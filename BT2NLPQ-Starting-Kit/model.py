import pickle
import random
import numpy as np
from collections import defaultdict, Counter

class InterpolatedLM:
    def __init__(self):
        self.unigrams = Counter()
        self.bigrams = defaultdict(Counter)
        self.trigrams = defaultdict(Counter)
        self.quadgrams = defaultdict(Counter)
        
        self.total_words = 0
        # Trọng số nội suy [Unigram, Bigram, Trigram, Quadgram]
        self.lambdas = [0.05, 0.15, 0.35, 0.45] 

    def fit(self, data):
        for line in data:
            # Thêm padding
            tokens = ['<START>', '<START>', '<START>'] + line.split() + ['<END>']
            self.total_words += len(tokens)
            
            for i in range(len(tokens)):
                # 1. Unigram
                self.unigrams[tokens[i]] += 1
                
                # 2. Bigram
                if i >= 1:
                    prev1 = tokens[i-1]
                    self.bigrams[prev1][tokens[i]] += 1
                
                # 3. Trigram
                if i >= 2:
                    prev2 = (tokens[i-2], tokens[i-1])
                    self.trigrams[prev2][tokens[i]] += 1
                    
                # 4. Quadgram
                if i >= 3:
                    prev3 = (tokens[i-3], tokens[i-2], tokens[i-1])
                    self.quadgrams[prev3][tokens[i]] += 1

    def _get_proba(self, word, history):
        h_bigram = history[-1]
        h_trigram = tuple(history[-2:])
        h_quadgram = tuple(history[-3:])
        
        # P(w)
        p1 = self.unigrams[word] / self.total_words
        
        # P(w | w-1)
        bi_counts = self.bigrams.get(h_bigram, {})
        count_bi = bi_counts.get(word, 0)
        total_bi = sum(bi_counts.values())
        p2 = count_bi / total_bi if total_bi > 0 else 0
        
        # P(w | w-2, w-1)
        tri_counts = self.trigrams.get(h_trigram, {})
        count_tri = tri_counts.get(word, 0)
        total_tri = sum(tri_counts.values())
        p3 = count_tri / total_tri if total_tri > 0 else 0
        
        # P(w | w-3, w-2, w-1)
        quad_counts = self.quadgrams.get(h_quadgram, {})
        count_quad = quad_counts.get(word, 0)
        total_quad = sum(quad_counts.values())
        p4 = count_quad / total_quad if total_quad > 0 else 0
        
        final_prob = (self.lambdas[0] * p1 + 
                      self.lambdas[1] * p2 + 
                      self.lambdas[2] * p3 + 
                      self.lambdas[3] * p4)
        return final_prob



class FirstLM(InterpolatedLM):
    def save(self):
        self.bigrams = dict(self.bigrams)
        self.trigrams = dict(self.trigrams)
        self.quadgrams = dict(self.quadgrams)
        pickle.dump(self, open("FirstLM.mdl", "wb"))
    def generate(self):
        current_history = ['<START>', '<START>', '<START>']
        generated_sentence = []
        MAX_LEN = 50
        
        for _ in range(MAX_LEN):
            candidates = set()
            
            h1 = current_history[-1]
            h2 = tuple(current_history[-2:])
            h3 = tuple(current_history[-3:])
            
            # Lấy các từ ứng viên từ lịch sử
            if h1 in self.bigrams:
                candidates.update(self.bigrams[h1].keys())
            if h2 in self.trigrams:
                candidates.update(self.trigrams[h2].keys())
            if h3 in self.quadgrams:
                candidates.update(self.quadgrams[h3].keys())
            
            # --- FIX: LOẠI BỎ THẺ <START> KHỎI ỨNG VIÊN ---
            if '<START>' in candidates:
                candidates.remove('<START>')
            
            # Nếu không có ứng viên (hoặc chỉ có START), lấy top unigram
            if not candidates:
                # Lấy 50 từ phổ biến nhất nhưng loại trừ <START>
                top_words = [w for w, c in self.unigrams.most_common(60) if w != '<START>']
                candidates.update(top_words[:50])
            
            candidate_list = list(candidates)
            probs = []
            
            for word in candidate_list:
                probs.append(self._get_proba(word, current_history))
            
            probs = np.array(probs)
            
            # Chuẩn hóa xác suất (tránh lỗi chia cho 0 nếu xác suất quá nhỏ)
            if probs.sum() == 0:
                probs = np.ones(len(probs)) / len(probs)
            else:
                probs = probs / probs.sum()
            
            next_word = np.random.choice(candidate_list, p=probs)
            next_word = str(next_word)
            
            if next_word == '<END>':
                break
                
            generated_sentence.append(next_word)
            current_history.append(next_word)
        
        return " ".join(generated_sentence) + " oh no i am very tired"


class SecondLM(InterpolatedLM):
    def save(self):
        self.bigrams = dict(self.bigrams)
        self.trigrams = dict(self.trigrams)
        self.quadgrams = dict(self.quadgrams)
        pickle.dump(self, open("SecondLM.mdl", "wb"))
    def generate(self):
        current_history = ['<START>', '<START>', '<START>']
        generated_sentence = []
        MAX_LEN = 50
        
        for _ in range(MAX_LEN):
            candidates = set()
            
            h1 = current_history[-1]
            h2 = tuple(current_history[-2:])
            h3 = tuple(current_history[-3:])
            
            # Lấy các từ ứng viên từ lịch sử
            if h1 in self.bigrams:
                candidates.update(self.bigrams[h1].keys())
            if h2 in self.trigrams:
                candidates.update(self.trigrams[h2].keys())
            if h3 in self.quadgrams:
                candidates.update(self.quadgrams[h3].keys())
            
            # --- FIX: LOẠI BỎ THẺ <START> KHỎI ỨNG VIÊN ---
            if '<START>' in candidates:
                candidates.remove('<START>')
            
            # Nếu không có ứng viên (hoặc chỉ có START), lấy top unigram
            if not candidates:
                # Lấy 50 từ phổ biến nhất nhưng loại trừ <START>
                top_words = [w for w, c in self.unigrams.most_common(60) if w != '<START>']
                candidates.update(top_words[:50])
            
            candidate_list = list(candidates)
            probs = []
            
            for word in candidate_list:
                probs.append(self._get_proba(word, current_history))
            
            probs = np.array(probs)
            
            # Chuẩn hóa xác suất (tránh lỗi chia cho 0 nếu xác suất quá nhỏ)
            if probs.sum() == 0:
                probs = np.ones(len(probs)) / len(probs)
            else:
                probs = probs / probs.sum()
            
            next_word = np.random.choice(candidate_list, p=probs)
            next_word = str(next_word)
            
            if next_word == '<END>':
                break
                
            generated_sentence.append(next_word)
            current_history.append(next_word)
        
        return " ".join(generated_sentence) + " haha, I am so happy, everything is good"