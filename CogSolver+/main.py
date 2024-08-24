# coding: utf-8
import os
from src.config import args
os.environ['CUDA_VISIBLE_DEVICES']=args.cuda_id
from src.train_and_evaluate import *
from src.rehearsal import *
from src.models import *
import time
import torch.optim
from src.expressions_transfer import *
import logging
import random
import numpy as np

batch_size = 4
hidden_size = 512
n_epochs = 40
learning_rate = 0.0008
bert_lr = 8e-6
weight_decay = 1e-5
beam_size = 5
n_layers = 2
embedding_size = 768
dropout = 0.5
BertTokenizer_path = "bert_model/bert-base-uncased" #本地修改了词表
Bert_model_path = "bert-base-uncased"    #"bert-base-uncased"#在线加载
best_acc = []
word_threshold = 0.7
op_threshold = 0.5
recursion_depth = 5
r = 10
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", filename='log')

for fold in range(5):
    train_data_path = "./data/fold"+str(fold)+"/train.json"
    test_data_path = "./data/fold"+str(fold)+"/dev.json"
    train_data = load_raw_data1(train_data_path)
    test_data = load_raw_data1(test_data_path)
    pairs1, generate_nums1, copy_nums1 = transfer_num1(train_data,use_bert_flag=True,model_name=BertTokenizer_path)
    pairs2, _ , _ = transfer_num1(test_data,use_bert_flag=True,model_name=BertTokenizer_path)
    pairs_trained = pairs1
    pairs_tested = pairs2
    input_lang, output_lang, train_pairs, test_pairs, _ = prepare_data(pairs_trained, pairs_tested, pairs_tested, 5, generate_nums1,copy_nums1, tree=True)

    # Initialize models
    embedding = Embedding(input_size=input_lang.n_words, embedding_size=embedding_size, dropout=0.5)
    # embedding = BertEncoder(Bert_model_path,dropout=dropout)
    encoder = EncoderSeq(embedding_size=768 , hidden_size=hidden_size,n_layers=n_layers,dropout=dropout)
    predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums1 - 1 - len(generate_nums1),
                            input_size=len(generate_nums1))
    generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums1 - 1 - len(generate_nums1),
                            embedding_size=embedding_size)
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
    knowledge_base = Knowledge_Base(embed_model=embedding, op_emb=predict.ops, vocab_dict=input_lang.word2index, word_threshold=word_threshold, op_threshold=op_threshold, fold=fold)

    embedding_optimizer = torch.optim.Adam(embedding.parameters(), lr=bert_lr, weight_decay=weight_decay)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)
    know_optimizer = torch.optim.AdamW(knowledge_base.parameters(), lr=learning_rate, weight_decay=weight_decay)

    embedding_scheduler = torch.optim.lr_scheduler.StepLR(embedding_optimizer, step_size=10, gamma=0.5)
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=10, gamma=0.5)
    predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=10, gamma=0.5)
    generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=10, gamma=0.5)
    merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=10, gamma=0.5)
    know_scheduler = torch.optim.lr_scheduler.StepLR(know_optimizer, step_size=30, gamma=0.5)

    # Move models to GPU
    if USE_CUDA:
        embedding.cuda()
        encoder.cuda()
        predict.cuda()
        generate.cuda()
        merge.cuda()
        knowledge_base.cuda()

    generate_num_ids = []
    for num in generate_nums1:
        generate_num_ids.append(output_lang.word2index[num])
    best_val_cor = 0
    best_eval_total  = 1
    
    bert_tokenizer = BertTokenizer.from_pretrained(BertTokenizer_path)
    
    for epoch in range(n_epochs):
        embedding_scheduler.step()
        encoder_scheduler.step()
        predict_scheduler.step()
        generate_scheduler.step()
        merge_scheduler.step()
        know_scheduler.step()
        loss_total = 0
        input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, input_ids, exer_sims = prepare_train_batch1(train_pairs, batch_size)
        logging.info(f"fold: {str(fold)}, epoch: {epoch + 1}")
        start = time.time()
        
        with torch.no_grad():
            word_word_old, word_op_old = knowledge_base.cal_know()
        time_dif = [0 for _ in range(len(train_pairs))]
        cur_time = 1
    
        for idx in range(len(input_lengths)):
            loss = train_tree(
                input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                num_stack_batches[idx], num_size_batches[idx], generate_num_ids, embedding,encoder, predict, generate, merge, knowledge_base,
                embedding_optimizer,encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, know_optimizer, input_lang,output_lang, num_pos_batches[idx])
            loss_total += loss
            
            batch_id = input_ids[idx]
            for _i in batch_id:
                time_dif[_i] = cur_time
            cur_time += 1
        
        #------------------
        # Knowledge Recall
        with torch.no_grad():
            word_word_new, word_op_new = knowledge_base.cal_know()
        time_dif = np.sqrt(time_dif)
        time_pro = time_dif / sum(time_dif)
        exer_sim = exer_sims[-1] * time_pro  # exercise similarity based on TF-IDF and time difference, size = (exer_num)

        for i in batch_id:
             exer_sim[i] = 0  # mask train_batch
        data_reh_id = random.choices(range(len(exer_sim)),weights=exer_sim,k=10)  # list, len = 10
    
        data_train = [input_batches[-1], input_lengths[-1], output_batches[-1], output_lengths[-1], nums_batches[-1],
                num_stack_batches[-1], num_pos_batches[-1], num_size_batches[-1]]
        data_reh = [train_pairs[id_] for id_ in data_reh_id]
        data_reh = [x[0] for x in prepare_train_batch1(data_reh, len(data_reh_id))[:-2]]
        models = (embedding, encoder, predict, generate, merge, knowledge_base)
        meta_term = (generate_num_ids, input_lang, output_lang)
            
        word_word_old.requires_grad = True
        word_op_old.requires_grad = True
        influence_ww, influence_wo = rehearsal_know(data_train, data_reh, word_word_old, word_op_old, models, meta_term, recursion_depth, r)
    
        delta = 1e-9
        ww_inc = (influence_ww * (word_word_new-word_word_old)<=-delta)
        wo_inc = (influence_wo * (word_op_new-word_op_old)<=-delta)
        ww_dec = (influence_ww * (word_word_new-word_word_old)>=delta)
        wo_dec = (influence_wo * (word_op_new-word_op_old)>=delta)
        
        batch_text = [bert_tokenizer.convert_ids_to_tokens(train_pairs[id_][0]) for id_ in data_reh_id]
        inverse = {value:key for (key, value) in input_lang.word2index.items()}
        ww_where = torch.where(ww_inc)
        for i in range(len(ww_where[0])):
            x_id = int(ww_where[0][i])
            y_id = int(ww_where[1][i])
            x_w = inverse[x_id]
            y_w = inverse[y_id]
            
        ww_where = torch.where(ww_dec)
        for i in range(len(ww_where[0])):
            x_id = int(ww_where[0][i])
            y_id = int(ww_where[1][i])
            x_w = inverse[x_id]
            y_w = inverse[y_id]
            
        wo_where = torch.where(wo_inc)
        op_inverse = list(set("*/+-^"))
        for i in range(len(wo_where[0])):
            x_id = int(wo_where[0][i])
            y_id = int(wo_where[1][i])
            x_w = inverse[x_id]
            y_w = op_inverse[y_id]
            
        wo_where = torch.where(wo_dec)
        for i in range(len(wo_where[0])):
            x_id = int(wo_where[0][i])
            y_id = int(wo_where[1][i])
            x_w = inverse[x_id]
            y_w = op_inverse[y_id]
                
        gamma = 0.1
        delta_word_word = (ww_dec * gamma + ww_inc * (1+gamma) + ~ww_dec * ~ww_inc) * word_word_new - (ww_dec * (gamma-1) + ww_inc * gamma + ww_dec * ww_inc) * word_word_old - word_word_new
        delta_word_op = (wo_dec * gamma + wo_inc * (1+gamma) + ~wo_dec * ~wo_inc) * word_op_new - (wo_dec * (gamma-1) + wo_inc * gamma + wo_dec * wo_inc) * word_op_old - word_op_new
        knowledge_base.recall_know(delta_word_word.detach(), delta_word_op.detach())
        #------------------

        logging.info(f"loss: {loss_total / len(input_lengths)}")
        logging.info(f"training time: {time_since(time.time() - start)}")
        # print("--------------------------------")
        #开始valid
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        start = time.time()
        #pairs:input_seq, len(input_seq), out_seq(prefix with index), len(out_seq), nums, num_pos, num_stack
        for test_batch in test_pairs:
            test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids,embedding,encoder, predict, generate,
                                        merge, knowledge_base, input_lang,output_lang, test_batch[5] , beam_size=beam_size)
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
            if val_ac:
                value_ac += 1
            else:
                if epoch == n_epochs-1:
                    text = bert_tokenizer.convert_ids_to_tokens(test_batch[0])
                    gt = [output_lang.index2word[i] for i in test_batch[2]]
                    pre = [output_lang.index2word[i] for i in test_res]
                    print("text:", text)
                    print("gt:", gt)
                    print("pre:", pre)
            if equ_ac:
                equation_ac += 1
            eval_total += 1
        # print(equation_ac, value_ac, eval_total)

        logging.info(f"valid_eq_acc: {float(equation_ac) / eval_total}, valid_an_acc: {float(value_ac) / eval_total}")
        logging.info(f"time: {time_since(time.time() - start)}")
        # print("------------------------------------------------------")
        if float(value_ac) / eval_total > best_val_cor / best_eval_total:
            best_val_cor = value_ac
            best_eval_total = eval_total
            torch.save(encoder.state_dict(), "models/encoder_"+str(fold))
            torch.save(predict.state_dict(), "models/predict_"+str(fold))
            torch.save(generate.state_dict(), "models/generate_"+str(fold))
            torch.save(merge.state_dict(), "models/merge_"+str(fold))
            torch.save(knowledge_base.state_dict(), "models/knowledge_base_"+str(fold))
    
    best_acc.append((best_val_cor,best_eval_total))     
            

# 开始测试
total_value_corr = 0
total_len = 0
folds_scores=[]
for w in range(len(best_acc)):
    folds_scores.append(float(best_acc[w][0])/best_acc[w][1])
    total_value_corr += best_acc[w][0]
    total_len += best_acc[w][1]
fold_acc_score = float(total_value_corr)/total_len
print("fold0-fold4 value accs: ",folds_scores)
print("final Val score: ",fold_acc_score)

