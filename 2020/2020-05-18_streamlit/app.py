import streamlit as st
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def sentiment_analyzer_scores(sentence, analyser, translator):
   trans = translator.translate(sentence).text
   score = analyser.polarity_scores(trans)

   score = score['compound']
   if score >= 0.05:
      return 'The sentiment of your text is Positive'
   elif score > -0.5 and score < 0.05:
      return 'The sentiment of your text is Neutral'
   else:
      return 'The sentiment of your text is Negative'


def main():
    analyser = SentimentIntensityAnalyzer()
    translator = Translator()

    st.markdown('''
    # Markdown header.

    [Markdown link](https://github.com).
    ''')

    sentence = st.text_area('Write your sentence')

    if st.button('Submit'):
        result = sentiment_analyzer_scores(sentence, analyser, translator)
        st.balloons()
        st.success(result)

if __name__ == '__main__':
    main()
