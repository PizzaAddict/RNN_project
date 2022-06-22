from collections import Counter
import re, collections
from tqdm.auto import tqdm
from os.path import exists 
import pickle
ENDWORD_TOKEN = '</w>'

def get_stats(dictionary):#dic: whole dic. 문서의 모든 단어 전체
    # 유니그램의 pair들의 빈도수를 카운트
    pairs = collections.defaultdict(int)
    for word, freq in dictionary.items():
        symbols = word.split()#whitespace를 기준으로 나눈다. 때문에 사전에 분리가 되어있어야 함
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq #tuple 이 key가 된다.
    return pairs

def merge_dictionary(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))# 묶고싶은 unigram은 공백으로 떨어져 있을 것이므로...
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)#word 안에서 pattern에 맞는 부분을 join 해서 교체한다
        v_out[w_out] = v_in[word]#그걸 다시 key로 등록
    return v_out

def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as a tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Vocabulary:
    #byte-pair encoded vocab
    def __init__(self, document, num_merges = 10000):
        if exists('./vocab.pickle'):
            with open('./vocab.pickle','rb') as file:
                data = pickle.load(file)
            self.num_merges, self.itow, self.wtoi, self. wtoi_reverse = data
        else:
            self.num_merges = num_merges
            self.itow = {0: '<PAD>',1: '<UNK>',2: ENDWORD_TOKEN,}
            self.wtoi = {w:i for i, w in self.itow.items()}
            self.wtoi_reverse = {'<PAD>':'<PAD>', '<UNK>':'<UNK>', ENDWORD_TOKEN:ENDWORD_TOKEN,}
            self.build_vocabulary(document)
            wtoi_r_r = {t:w for w, t in self.wtoi_reverse.items()}
            self.itow = {i:wtoi_r_r[w] for w, i in self.wtoi.items()}
            with open('./vocab.pickle','wb') as file:
                    data = pickle.dump((self.num_merges, self.itow, self.wtoi, self. wtoi_reverse),file)

    def __len__(self):
        return len(self.itow)

    def spacer(self, text):
        return [tok.lower().strip() for tok in text.split(' ')]#무조건 lowercase로만 작동

    def build_vocabulary(self, sentences,):#sentence: pandas로 불러온뒤 text 줄만...
        to_wtoi, to_wtoi_r = self.count_characters(sentences)
        self.wtoi.update(to_wtoi)
        self.wtoi_reverse.update(to_wtoi_r)
        #여기까지 문서 안의 모든 문자 추가 완료
        word_list = []
        for sentence in sentences:
            for word in self.spacer(sentence):
                if word:
                    word_list.append(' '.join(tuple(word))+ENDWORD_TOKEN)
                    
        document_freq = dict(Counter(word_list))
        self.generate_bpe_vocab(document_freq,self.num_merges)

    def generate_bpe_vocab(self,dictionary, num_merges):
        value_max = max(self.wtoi.values())
        for i in tqdm(range(num_merges)):
            pairs = get_stats(dictionary)
            best = max(pairs, key=pairs.get)#문서 전체에서 가장 빈도수가 높은 pair
            dictionary = merge_dictionary(best, dictionary)

            self.wtoi[best] = i+value_max+1
            self.wtoi_reverse[best[0] + best[1]] = best

    def count_characters(self, sentences):
        value_max = max(self.wtoi.values())
        sentence_list = []
        for sentence in tqdm(sentences):
            sentence_list.append(sentence.lower())
        merged = list(tuple(''.join(sentence_list)))
        merged = sorted(list(set(merged)))
        merged_dict = {c:i for c,i in zip(merged,range(value_max+1,value_max+len(merged)+1))}
        merged_dict_reverse = {i:i for i,v in merged_dict.items()}
        return merged_dict, merged_dict_reverse

    def encode(self, orig):#bpe에 의거해서 주어진 단일 word를 잘라서 출력

        word = tuple(orig) + (ENDWORD_TOKEN,)#단어의 끝에 존재하는 unigram인 경우를 체크
        #display(Markdown("__word split into characters:__ <tt>{}</tt>".format(word)))

        pairs = get_pairs(word)

        if not pairs:
            return orig

        iteration = 0
        while True:
            iteration += 1
            #display(Markdown("__Iteration {}:__".format(iteration)))

            bigram = min(pairs, key = lambda pair: self.wtoi.get(pair, float('inf')))
            if bigram not in self.wtoi:
                #display(Markdown("__Candidate not in BPE merges, algorithm stops.__"))
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        """
        # 특별 토큰인 </w>는 출력하지 않는다.
        if word[-1] == ENDWORD_TOKEN:
            word = word[:-1]

        elif word[-1].endswith(ENDWORD_TOKEN):#Unigram 자체가 /w를 포함하는 경우.
            word = word[:-1] + (word[-1].replace(ENDWORD_TOKEN,''),)
        """
        def return_index(string):
            try:
                return self.wtoi[self.wtoi_reverse[string]]
            except KeyError:
                return 1#존재하지 않는 단어
        return list(map(return_index,word))

    def __call__(self, string):
        tokenized = []
        spaced = self.spacer(string)
        for word in spaced:
            tokenized = tokenized + self.encode(word)
        return tokenized

    def itow_(self, tokenlist):
        return ''.join(list(map(lambda x: self.itow[x], tokenlist)))

"""
import pandas as pd
df = pd.read_csv('./Data/train.csv')
test_doc = df['text'].values.tolist()
test_vocab = Vocabulary(test_doc,1000)
for i in range(100):
    sentences = test_doc[i]
    tokenized = test_vocab(sentences)
    print(tokenized)
    print(test_vocab.itow_(tokenized))
"""