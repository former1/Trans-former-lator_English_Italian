import os
import numpy as np
import torch 
from transformers import AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_from_disk
from tqdm import tqdm

from model import Transformer,TransformerConfig
from data import TranslationCollator
from tokenizer import ItalianTokenizer

device = torch.device("cuda")

def main():
        
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


    embedding_dimension =512
    num_attention_heads=8
    attention_dropout_p =0.0
    hidden_dropout_p=0.0
    mlp_ratio=4
    encoder_depth =6
    decoder_depth =6
    max_src_len=512 
    max_tgt_len= 512
    learn_pos_embed = False

    tgt_tokenizer= ItalianTokenizer("trained_tokenizer/italian_wp.json")
    src_tokenizer=AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    #path_to_data="/Users/omer/Desktop/dataset_hf/tokenized_english2italian_corpus"
    path_to_data="/workspace/tokenized_english2italian_corpus"

    batch_size = 64
    gradient_acc_steps = 4
    #batch_size=4
    #gradient_acc_steps=2
    num_workers=4

    learning_rate=1e-4
    num_training_steps=50000
    num_warmup_steps=2000
    scheduler_type="cosine"
    evaluation_steps=5000
    bias_norm_weight_decay= False
    weight_decay=0.001
    betas=(0.9,0.98)
    adam_eps=1e-6

    working_directory="/workspace/work_dir"    
    experiment_name="Translate_IT_to_ENG"
    logging_interval=1

    resume_from_checkpoint=None


    # Train
    path_to_experiment=os.path.join(working_directory, experiment_name)
    os.makedirs(path_to_experiment, exist_ok=True)
    accelerator = Accelerator(project_dir=path_to_experiment, mixed_precision="fp16")

    #accelerator.init_trackers(experiment_name)
    config= TransformerConfig(embedding_dimension=embedding_dimension,
                            num_attention_heads=num_attention_heads,
                            attention_dropout_p=attention_dropout_p,
                            hidden_dropout_p=hidden_dropout_p,
                            mlp_ratio=mlp_ratio,
                            encoder_depth=encoder_depth,
                            decoder_depth=decoder_depth,
                            src_vocab_size=src_tokenizer.vocab_size,
                            tgt_vocab_size=tgt_tokenizer.vocab_size,
                            max_src_len=max_src_len,
                            max_tgt_len=max_tgt_len,
                            learn_pos_embed=learn_pos_embed
                            )



    # dataloaders
    dataset=load_from_disk(path_to_data)
    accelerator.print(dataset)

    collate_fn=TranslationCollator(src_tokenizer, tgt_tokenizer)

    minibatch_size =batch_size // gradient_acc_steps
    trainloader=DataLoader(dataset["train"], 
                        batch_size=minibatch_size, 
                        num_workers=num_workers,
                        collate_fn=collate_fn,
                        shuffle=True)

    testloader=DataLoader(dataset["test"], 
                        batch_size=minibatch_size, 
                        collate_fn=collate_fn,shuffle=False)

    # Prepare Model
    model= Transformer(config)

    total=0
    for param in model.parameters():
        if param.requires_grad:
            total +=param.numel()
    accelerator.print("number of params:", total)

    # Prepare Optimizer
    optimizer=torch.optim.AdamW(model.parameters(),
                                lr=learning_rate,
                                betas=betas,
                                eps=adam_eps)

    # scheduler
    scheduler=get_scheduler(name=scheduler_type,
                            optimizer=optimizer,
                            num_warmup_steps=num_warmup_steps * accelerator.num_processes,
                            num_training_steps=num_training_steps* accelerator.num_processes)
    #loss
    loss_fn=torch.nn.CrossEntropyLoss()


    # define sample for testing
    src_ids=torch.tensor(src_tokenizer("I love Aylin")["input_ids"]).unsqueeze(0)

    # prepare everything

    model, optimizer, trainloader, testloader, scheduler= accelerator.prepare(model,optimizer,trainloader, testloader,scheduler)

    accelerator.register_for_checkpointing(scheduler)

    # resume from checkpoint



    if resume_from_checkpoint is not None:
        path_to_checkpoint=os.path.join(path_to_experiment,resume_from_checkpoint)
        with accelerator.main_process_first():
            accelerator.load_state(path_to_checkpoint)

        completed_steps=int(resume_from_checkpoint.split("_")[-1])
        accelerator.print(f"resuming from iteration:",completed_steps)
    else:
        completed_steps=0


    train = True
    progress_bar= tqdm(range(completed_steps,num_training_steps), disable=not accelerator.is_main_process)

    while train:
        accumulate_steps=0
        accumulate_loss= 0
        accuracy=0

        for batch in trainloader:
            src_input_ids= batch["src_input_ids"].to(accelerator.device)
            src_pad_mask= batch["src_pad_mask"].to(accelerator.device)
            tgt_input_ids= batch["tgt_input_ids"].to(accelerator.device)
            tgt_pad_mask= batch["tgt_pad_mask"].to(accelerator.device)
            tgt_outputs= batch["tgt_outputs"].to(accelerator.device)

            output = model(src_input_ids,
                        tgt_input_ids,
                        src_pad_mask,
                        tgt_pad_mask)
            
            output=output.flatten(0,1)
            tgt_outputs=tgt_outputs.flatten()

            loss= loss_fn(output,tgt_outputs)

            loss= loss/ gradient_acc_steps
            accumulate_loss+= loss

            accelerator.backward(loss)

            output= output.argmax(axis=-1)
            mask=(tgt_outputs != -100)
            output= output[mask]
            tgt_outputs=tgt_outputs[mask]
            acc = (output == tgt_outputs).sum() / len(output)
            accuracy +=  acc/ gradient_acc_steps
            accumulate_steps +=1



            if accumulate_steps %gradient_acc_steps== 0:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0     )
                optimizer.step()
                optimizer.zero_grad(set_to_none= True)
                scheduler.step()


                if accumulate_steps % logging_interval ==0:
                    accumulate_loss= accumulate_loss.detach()
                    accuracy= accuracy.detach()

                    if accelerator.num_processes >1:
                        accumulate_loss=torch.mean(accelerator.gather_for_metrics(accumulate_loss))
                        accuracy = torch.mean(accelerator.gather_for_metrics(accuracy))

                    log= {"train_loss": accumulate_loss,
                          "trainin_acc":accuracy,
                          "learning_rate": scheduler.get_last_lr()[0]}
                    
                    accelerator.log(log,step=completed_steps)

                    logging_string = f"[{completed_steps}/{num_training_steps}] Training Loss:{accumulate_loss} | Training Acc: {accuracy}"

                    if accelerator.is_main_process:
                        progress_bar.write(logging_string)

                if completed_steps % evaluation_steps == 0:

                    model.eval()
                    
                    print("Evaluating!")

                    test_losses = []
                    test_accs = []

                    for batch in tqdm(testloader, disable=not accelerator.is_main_process):

                        src_input_ids = batch["src_input_ids"].to(accelerator.device)
                        src_pad_mask = batch["src_pad_mask"].to(accelerator.device)
                        tgt_input_ids = batch["tgt_input_ids"].to(accelerator.device)
                        tgt_pad_mask = batch["tgt_pad_mask"].to(accelerator.device)
                        tgt_outputs = batch["tgt_outputs"].to(accelerator.device)

                        with torch.inference_mode():
                            output = model(src_input_ids, 
                                        tgt_input_ids, 
                                        src_pad_mask, 
                                        tgt_pad_mask)
                        
                        output = output.flatten(0,1)
                        tgt_outputs = tgt_outputs.flatten()
                        
                        loss = loss_fn(output, tgt_outputs)

                        output = output.argmax(axis=-1)
                        mask = (tgt_outputs != -100)
                        output = output[mask]
                        tgt_outputs = tgt_outputs[mask]
                        accuracy = (output == tgt_outputs).sum() / len(output)   

                        loss = loss.detach()
                        accuracy = accuracy.detach()

                        if accelerator.num_processes > 1:
                            loss = torch.mean(accelerator.gather_for_metrics(loss))
                            accuracy = torch.mean(accelerator.gather_for_metrics(accuracy))
                
                        test_losses.append(loss.item())
                        test_accs.append(accuracy.item())

                    test_loss = np.mean(test_losses)
                    test_acc = np.mean(test_accs)

                    log = {"test_loss": test_loss,
                            "test_acc": test_acc}   
                    
                    logging_string = f"Testing Loss: {test_loss} | Testing Acc: {test_acc}"
                    if accelerator.is_main_process:
                        progress_bar.write(logging_string)
                    
                    accelerator.log(log, step=completed_steps)
                    
                    # delete previous checkpoint to save space
                    import shutil
                    for item in os.listdir(path_to_experiment):
                        if item.startswith("checkpoint_"):
                            shutil.rmtree(os.path.join(path_to_experiment, item))
                    
                    accelerator.save_state(os.path.join(path_to_experiment, f"checkpoint_{completed_steps}"))

                    # testing sentence
                    if accelerator.is_main_process:
                        src_ids = src_ids.to(accelerator.device)
                        unwrapped= accelerator.unwrap_model(model)
                        translated= unwrapped.inference(src_ids,
                                                        tgt_start_id=tgt_tokenizer.special_tokens_dict["[BOS]"],
                                                        tgt_end_id=tgt_tokenizer.special_tokens_dict["[EOS]"])
 

                        translated= tgt_tokenizer.decode(translated, skip_special_tokens=False)
                        accelerator.print(translated)
                        #if accelerator.is_main_process:
                        #    progress_bar.write(f"Translation:",translated)

                    model.train()
                if completed_steps >=num_training_steps:
                    train= False
                    accelerator.save_state(os.path.join(path_to_experiment,"final_checkpoint"))
                    break
                
                completed_steps +=1
                progress_bar.update(1)
                accumulate_loss=0
                accuracy=0

    accelerator.end_training()

  
if __name__ == "__main__":
    main()