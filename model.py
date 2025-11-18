'''
Sample predictive model.
You must supply at least 2 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
'''
import numpy as np   
import nltk
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

#nltk.download('punkt')
#nltk.download("stopwords")
# Tải các gói dữ liệu cần thiết
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

def get_wordnet_pos(word):
    """
    Hàm phụ trợ: Map POS tag của NLTK sang format của WordNetLemmatizer
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    
    # Mặc định là Danh từ (NOUN) nếu không tìm thấy
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text(text):
    """
    Hàm chính: Nhận vào một câu (string) và trả về câu đã lemmatize (string)
    """
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    
    result = []
    for word in tokens:
        # Lấy từ loại và lemmatize
        lemma = lemmatizer.lemmatize(word, get_wordnet_pos(word))
        result.append(lemma)
    
    # Nối lại thành chuỗi để đưa vào Vectorizer
    return " ".join(result)


 # ------- model definition --------   
class model:
  def __init__(self):
    self.classifier = None
    self.vectorizer = None
    
  def preprocess(self, X_data):
    # X_data là một list các chuỗi văn bản
    # Áp dụng hàm lemmatize_text cho từng dòng
    return [lemmatize_text(text) for text in X_data]

  def fit(self, XTrain, YTrain):
    XTrain = self.preprocess(XTrain)
    self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2), # Dùng cụm 1 và 2 từ
            min_df=3            # Lọc nhiễu: từ phải xuất hiện ít nhất 3 lần
        )
    vTrain = self.vectorizer.fit_transform(XTrain).toarray()
    self.classifier = LogisticRegression(
            solver='liblinear',
            random_state=42,
            C=5.0,              # Tăng C
            # !!! QUAN TRỌNG: Xử lý mất cân bằng nhãn
            class_weight='balanced' 
        )
    self.classifier.fit(vTrain, YTrain)

  def predict(self, XTest):	
    vTest = self.vectorizer.transform(XTest).toarray()
    YTest = self.classifier.predict(vTest)
    return YTest
