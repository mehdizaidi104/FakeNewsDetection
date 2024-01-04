import streamlit as st
import pickle
import re #for searching words in a text or paragraph
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.image as mpimg
port_stem = PorterStemmer()
vectorizer = TfidfVectorizer()
model = LogisticRegression()

# Load the trained model
with open('TrainedModel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the vectorizer
with open('TrainedVectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


#We are creating a function stemming which will take text as input and it'll perform all the operations and return a
#string stemmed_content
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content) #In this step we are calling the regular expression library.
    #This will only include lower and upper case alphabets and remove anything i.e. not alphabet by a space i.e. ' '
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    #In this step we are taking each word and performing the stemming operation of each indivudual word. Basically in this
    #step we are removing all the stop words
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content




# Function to preprocess and predict fake news
def predictFakeNews(news):
    processedNews = stemming(news)
    vectorizedNews = vectorizer.transform([processedNews])
    prediction = model.predict(vectorizedNews)
    return prediction[0]  # Return the prediction directly
    
if __name__ == '__main__':
    st.title('Fake News Detection System')
    st.subheader('Input the news content below')
    sentence = st.text_area("Enter the News ↓", "Some News")
    predict_btt = st.button('Predict') #Returns True if button is clicked else returns False
    if predict_btt==True:
        prediction_class = predictFakeNews(sentence)
        if prediction_class == 1:
            st.warning('Unreliable')
        else:
            st.success('Reliable')
    #Creating a Checkbox to display the Graph   
    cb = st.checkbox('Accuracy Score Graph')
    if cb==True:
        image = mpimg.imread('\\Users\\Acer\\Downloads\\graph.png')
        st.image(image, caption='Accuracy Score Comparison')
    else:
        st.write('Graph Hidden')
