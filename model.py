'''
Sample predictive model.
You must supply at least 2 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
'''
import numpy as np   
import nltk
import re
import html
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
########## Classifiers
#import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

#nltk.download('punkt')
#nltk.download("stopwords")
# Tải các gói dữ liệu cần thiết
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
#nltk.download('punkt_tab')
#nltk.download('averaged_perceptron_tagger_eng')

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

def clean_by_majority_vote(x, y):
    """
    Lọc trùng lặp bằng cách giữ lại nhãn xuất hiện nhiều nhất cho mỗi text.
    """
    df = pd.DataFrame({'text':x, 'label': y})
    counts = df.groupby(['text', 'label']).size().reset_index(name='count')
    counts = counts.sort_values(['text', 'count'], ascending=[True, False])
    
    df_deduped = counts.drop_duplicates(subset=['text'], keep='first')
    return df_deduped[['text', 'label']]

 # ------- model definition --------   
class model1:
  def __init__(self):
    self.classifier = None
    self.vectorizer = None
    
  def preprocess_text(self, text):
    """
    Hàm làm sạch từng dòng dữ liệu cụ thể cho bộ data này
    """
    # 1. QUAN TRỌNG NHẤT: Giải mã HTML entities
    # Dữ liệu của bạn chứa: '&#128514;' -> convert thành '😂'
    # Nếu không làm bước này, máy chỉ thấy chuỗi ký tự vô nghĩa.
    text = html.unescape(str(text))
    text = text.replace(":", "").replace("_", " ")

    # 2. Chuyển về chữ thường
    text = text.lower()

    # 3. Xóa các Mentions vô nghĩa (ví dụ: @T_Madison_x:, @__BrighterDays:)
    # Chúng ta không muốn model học thuộc lòng tên người dùng.
    text = re.sub(r'@[A-Za-z0-9_]+:?', '', text)

    # 4. Xóa ký hiệu Retweet (RT) xuất hiện dày đặc đầu câu
    text = re.sub(r'\brt\b', '', text)

    # 5. Xóa URL (nếu có)
    text = re.sub(r'http\S+', '', text)

    # 6. (Tùy chọn) Xóa bớt dấu chấm than/hỏi dư thừa nhưng giữ lại 1 cái
    # !!! -> !
    text = re.sub(r'!+', '!', text)
    text = re.sub(r'\?+', '?', text)

    return text.strip()

  def fit(self, XTrain, YTrain):
    df = clean_by_majority_vote(XTrain, YTrain)
    XTrain = df['text']
    YTrain = df['label']
    XTrain = [self.preprocess_text(x) for x in XTrain]

    print(XTrain)
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
            max_iter=1000
            #C=5,              # Tăng C
            # !!! QUAN TRỌNG: Xử lý mất cân bằng nhãn
            #class_weight='balanced' 
        )
    self.classifier.fit(vTrain, YTrain)

  def predict(self, XTest):	
    vTest = self.vectorizer.transform(XTest).toarray()
    YTest = self.classifier.predict(vTest)
    return YTest

class model:
  def __init__(self):
    self.classifier = None
    self.vectorizer = None

  def preprocess_text(self, text):
    """
    Hàm làm sạch từng dòng dữ liệu cụ thể cho bộ data này
    """
    # 1. QUAN TRỌNG NHẤT: Giải mã HTML entities
    # Dữ liệu của bạn chứa: '&#128514;' -> convert thành '😂'
    # Nếu không làm bước này, máy chỉ thấy chuỗi ký tự vô nghĩa.
    text = html.unescape(str(text))
    text = text.replace(":", "").replace("_", " ")
    # 2. Chuyển về chữ thường
    text = text.lower()

    # 3. Xóa các Mentions vô nghĩa (ví dụ: @T_Madison_x:, @__BrighterDays:)
    # Chúng ta không muốn model học thuộc lòng tên người dùng.
    text = re.sub(r'@[A-Za-z0-9_]+:?', '', text)

    # 4. Xóa ký hiệu Retweet (RT) xuất hiện dày đặc đầu câu
    text = re.sub(r'\brt\b', '', text)

    # 5. Xóa URL (nếu có)
    text = re.sub(r'http\S+', '', text)

    # 6. (Tùy chọn) Xóa bớt dấu chấm than/hỏi dư thừa nhưng giữ lại 1 cái
    # !!! -> !
    text = re.sub(r'!+', '!', text)
    text = re.sub(r'\?+', '?', text)

    return text.strip()

  def fit(self, XTrain, YTrain):
    df = clean_by_majority_vote(XTrain, YTrain)
    XTrain = df['text']
    YTrain = df['label']
    XTrain = [self.preprocess_text(x) for x in XTrain]

    self.vectorizer = CountVectorizer()
    vTrain = self.vectorizer.fit_transform(XTrain).toarray()
    #self.classifier = MultinomialNB() 
    #self.classifier = LinearSVC(class_weight='balanced', dual=False) # 0.71
    #self.classifier = xgb.XGBClassifier(n_estimators=100)     # no module
    #self.classifier = RandomForestClassifier(class_weight='balanced', n_jobs=-1)
    self.classifier = LogisticRegression(
      solver='liblinear',
      random_state=42,
      max_iter=1000,
            #C=5,              # Tăng C
            # !!! QUAN TRỌNG: Xử lý mất cân bằng nhãn
      class_weight='balanced' 
    )
    self.classifier.fit(vTrain, YTrain)


  def predict(self, XTest):	
    vTest = self.vectorizer.transform(XTest).toarray()
    YTest = self.classifier.predict(vTest)
    return YTest
