from colorama import Fore, Back, Style
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
import sys, os
import pprint


import numpy as np
import torch
# from utils.language_model import get_optimizer_scheduler
# from utils.gpt2_args_parser import ArgsParser

from image_handler import Handler


def get_belief_new_dbsearch(sent):
    if '<|belief|>' in sent:
        tmp = sent.strip(' ').split('<|belief|>')[-1].split('<|endofbelief|>')[0]
    # elif 'belief.' in sent:
    #     tmp = sent.strip(' ').split('<belief>')[-1].split('<action>')[0]
    # elif 'belief' not in sent:
    #     return []
    else:
        return []
    # else:
    #     raise TypeError('unknown belief separator')
    tmp = tmp.strip(' .,')
    # assert tmp.endswith('<endofbelief>')
    tmp = tmp.replace('<|endofbelief|>', '')
    tmp = tmp.replace('<|endoftext|>', '')
    belief = tmp.split(',')
    new_belief = []
    for bs in belief:
        bs = bs.strip(' .,')
        if bs not in new_belief:
            new_belief.append(bs)
    return new_belief


def convert_belief(belief):
    dic = {}
    for bs in belief:
        if bs in [' ', '']:
            continue
        domain = bs.split(' ')[0]
        slot = bs.split(' ')[1]
        if slot == 'book':
            slot = ' '.join(bs.split(' ')[1:3])
            value = ' '.join(bs.split(' ')[3:])
        else:
            value = ' '.join(bs.split(' ')[2:])
        if domain not in dic:
            dic[domain] = {}
        try:
            dic[domain][slot] = value
        except:
            print(domain)
            print(slot)
    return dic

def get_turn_domain(beliefs, q):
  for k in beliefs.keys():
      if k not in q:
          q.append(k)
          turn_domain = k
          return turn_domain
  return q[-1]





def get_action_new(sent):
    if '<|action|>' not in sent:
        return []
    elif '<|belief|>' in sent:
        tmp = sent.split('<|belief|>')[-1].split('<|response|>')[0].split('<|action|>')[-1].strip()
    elif '<|action|>' in sent:
        tmp = sent.split('<|response|>')[0].split('<|action|>')[-1].strip()
    else:
        return []
    tmp = tmp.strip(' .,')
    # if not tmp.endswith('<endofaction>'):
    #     ipdb.set_trace()
    tmp = tmp.replace('<|endofaction|>', '')
    tmp = tmp.replace('<|endoftext|>', '')
    action = tmp.split(',')
    new_action = []
    for act in action:
        if act == '':
            continue
        act = act.strip(' .,')
        if act not in new_action:
            new_action.append(act)
    return new_action



def get_response_new(sent):
    if '<|response|>' in sent:
        tmp = sent.split('<|belief|>')[-1].split('<|action|>')[-1].split('<|response|>')[-1]
    else:
        return ''
    # if '<belief>' in sent:
    #     tmp = sent.split('<belief>')[-1].split('<action>')[-1].split('<response>')[-1]
    # elif '<action>' in sent:
    #     tmp = sent.split('<action>')[-1].split('<response>')[-1]
    # elif '<response>' in sent:
    #     tmp = sent.split('<response>')[-1]
    # else:
    #     tmp = sent
    tmp = tmp.strip(' .,')
    # assert tmp.endswith('<endofresponse>')
    tmp = tmp.replace('<|endofresponse|>', '')
    tmp = tmp.replace('<|endoftext|>', '')
    tokens = tokenizer.encode(tmp)
    new_tokens = []
    for tok in tokens:
        # if tok in break_tokens:
        if tok in tokenizer.encode(tokenizer.eos_token):
            continue
        new_tokens.append(tok)
    # ipdb.set_trace()
    response = tokenizer.decode(new_tokens).strip(' ,.')
    return response


def get_venuename(bs):
    name = ''
    if 'venuename' in bs[0]:
        tmp_list = bs[0].split('venuename')[-1].split(' ')
        #action = tmp_list[-1]
        name = ' '. join(tmp_list[:-1])
    return name


def get_open_span(bs):
    action_names = []
    for tmp in bs[0].split(';'):
        if 'open span' in tmp:
            action = tmp.split('open span')[-1].split(' ')[-1]
            name = tmp.split('open span')[-1].split(action)[0]
            action_names.append((name, action))
    return action_names





if __name__ == '__main__':

    img_handler_obj = Handler()
    #args = ArgsParser().parse()

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    #n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    pp = pprint.PrettyPrinter(indent=4)
    prev_beliefs = {}
    domain_queue = []

    #sys.stdout.flush()

    model_checkpoint = "./output/checkpoint-108420"

    decoding = "DECODING METHOD HERE"

    ## if decoding == 'nucleus':
    ##     TOP_P = float(sys.argv[3])

    delay = 0.5

    ## multiwoz_db = MultiWozDB()

    print('\nLoading Model', end="")

    if 'openai' in model_checkpoint:
        tokenizer = OpenAIGPTTokenizer.from_pretrained(model_checkpoint)
        model = OpenAIGPTLMHeadModel.from_pretrained(model_checkpoint)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)
        model = GPT2LMHeadModel.from_pretrained(model_checkpoint)

    # model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    model.to('cpu')

    break_tokens = tokenizer.encode(tokenizer.eos_token) + tokenizer.encode('?') + tokenizer.encode('!')
    # break_tokens = tokenizer.encode(tokenizer.eos_token)
    MAX_LEN = model.config.n_ctx

    if 'openai-gpt' in model_checkpoint:
        tokenizer.add_special_tokens({'bos_token': '<|endoftext|>'})
        tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})

    sample = 1
    print()
    print('\nWhat would you like to ask?')
    # history = []
    context = ''
    input_text = ''
    turn = 0
    # dbmatch = 0




    while True:
        print(Fore.GREEN)
        raw_text = input('You: ')
        print(Style.RESET_ALL)
        input_text = raw_text.replace('you> ', '')
        if input_text in ['q', 'quit']:
            break

        user = '<|user|> {}'.format(input_text)
        context = context + ' ' + user
        text = '<|endoftext|> <|context|> {} <|endofcontext|>'.format(context)

        #print(context)

        text = text.strip()
        indexed_tokens = tokenizer.encode(text)

        if len(indexed_tokens) > MAX_LEN:
            indexed_tokens = indexed_tokens[-1*MAX_LEN:]

        # Convert indexed tokens in a PyTorch tensor
        tokens_tensor = torch.tensor([indexed_tokens])

        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to('cpu')
        predicted_index = indexed_tokens[-1]

        # if decoding == 'nucleus':
        #     sample_output = model.generate(
        #         tokens_tensor,
        #         do_sample=True,
        #         max_length=MAX_LEN,
        #         top_p=TOP_P,
        #         top_k=0
        #     )
        # elif decoding == 'greedy':
        #     sample_output = model.generate(
        #         tokens_tensor,
        #         max_length=MAX_LEN,
        #         do_sample=False
        #     )
        # predicted_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)


        with torch.no_grad():
        # Greedy decoding

            while predicted_index not in break_tokens:
                outputs = model(tokens_tensor)
                predictions = outputs[0]
                predicted_index = torch.argmax(predictions[0, -1, :]).item()
                indexed_tokens += [predicted_index]
                tokens_tensor = torch.tensor([indexed_tokens]).to('cpu')
                if len(indexed_tokens) > MAX_LEN:
                    break
                if tokenizer.decode(indexed_tokens).endswith('<|endofbelief|>'):
                    break

        tmp_pred = tokenizer.decode(indexed_tokens)

        print('\ntmp_pred:\n', tmp_pred)




        belief_text = get_belief_new_dbsearch(tmp_pred)
        #print('\nbelief_text:\n', belief_text)


        beliefs = convert_belief(belief_text)
        # domain = list(beliefs.keys())[0]
        domain = get_turn_domain(beliefs, domain_queue)





        ###  QUERY HERE
        # if 'db' in model_checkpoint:
        #     if 'dbnmatch' in model_checkpoint:
        #         only_match = True
        #         db_text_tmp = get_db_text(beliefs, dom=domain, only_match=only_match)
        #     else:
        #         db_text_tmp = get_db_text(beliefs, dom=domain)
        #     db_text = ' <|dbsearch|> {} <|endofdbsearch|>'.format(' , '.join(db_text_tmp))
        #     text = tmp_pred + db_text
        # print(text)



        ####????

        # continue generation after creating db
        # indexed_tokens = tokenizer.encode(text)
        # if len(indexed_tokens) > MAX_LEN:
        #     indexed_tokens = indexed_tokens[-1 * MAX_LEN:]



        # Convert indexed tokens in a PyTorch tensor
        tokens_tensor = torch.tensor([indexed_tokens])

        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to('cpu')
        predicted_index = indexed_tokens[-1]

        truncate_action = False
        # Predict all tokens
        with torch.no_grad():
            while predicted_index not in break_tokens:
                outputs = model(tokens_tensor)
                predictions = outputs[0]
                predicted_index = torch.argmax(predictions[0, -1, :]).item()
                indexed_tokens += [predicted_index]
                if len(indexed_tokens) > MAX_LEN:
                    break


                predicted_text = tokenizer.decode(indexed_tokens)
                if '<|action|>' in predicted_text:
                    generated_actions = predicted_text.split('<|action|>')[-1].split('<|endofaction|>')[0].split(',')
                    new_actions = []
                    for a in generated_actions:
                        if a in ['', ' ']:
                            continue
                        new_actions.append(a.strip())
                    len_actions = len(new_actions)
                    if len(list(set(new_actions))) > len(new_actions) or (len_actions > 10 and not truncate_action):
                        # ipdb.set_trace()
                        actions = '<|action|> {} <|endofaction|>'.format(' , '.join(list(set(new_actions))))
                        indexed_tokens = tokenizer.encode('{} {}'.format(predicted_text.split('<|action|>')[0], actions))
                        # print('action truncated')
                        truncate_action = True
                tokens_tensor = torch.tensor([indexed_tokens]).to('cpu')

        predicted_text = tokenizer.decode(indexed_tokens)
        print('\npredicted_text:\n', predicted_text)

        action_text = get_action_new(predicted_text)
        #print('\naction_text:\n', action_text)
        response_text = get_response_new(predicted_text)
        #print('\nresponse_text:\n', response_text)
        # print(predicted_text)


        venuename = get_venuename(action_text)
        print('\nVenuename:\n',venuename )

        open_spans = get_open_span(action_text)
        print('\open_spans:\n', open_spans)


        # handling images

        if venuename:
            print(img_handler_obj.get_imgs_url(query = venuename +"in Singapore" , num_of_img = 5))

        # end of image handling


        # print(predicted_text)

        #db_results = multiwoz_db.queryResultVenues_new(domain, beliefs[domain], real_belief=True)

        # if domain == 'train':
        #     lex_response = lexicalize_train(response_text, db_results, beliefs, turn_domain=domain)
        # elif domain == 'hotel':
        #     lex_response = lexicalize_hotel(response_text, db_results, beliefs, turn_domain=domain)
        # else:
        #     ipdb.set_trace()
        #     raise TypeError('unknown domain')


        delex_system = '<|system|> {}'.format(response_text)
        context = context + ' ' + delex_system


        # delex_system = '<|system|> {}'.format(response_text)
        # system = '<|system|> {}'.format(lex_response)
        # context = context + ' ' + system


        print(Fore.CYAN + 'System: ', end="")
        for a in response_text.split(' '):
            print(a + ' ', end="")
            sys.stdout.flush()
            #time.sleep(delay)
        print(Style.RESET_ALL)
        #print(Fore.YELLOW + 'belief: {}'.format(beliefs) + Style.RESET_ALL)

        print(Style.RESET_ALL)

        turn += 1
        prev_beliefs = beliefs
