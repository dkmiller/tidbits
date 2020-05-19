import streamlit as st
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Tuple


def sentiment_score(sentence: str, analyser, translator) -> float:
   '''
   Calculate the sentiment of a sentance.
   '''
   translation = translator.translate(sentence).text
   score = analyser.polarity_scores(translation)

   return score['compound']


def sentiment(score: float) -> str:
   '''
   Convert a sentiment score into a text representation.
   '''
   if score >= 0.05:
      return 'Positive'
   elif abs(score) < 0.05:
      return 'Neutral'
   else:
      return 'Negative'


def sentiment_analyzer_scores(sentence, analyser, translator) -> Tuple[float, str]:
   '''
   Return sentiment score, together with a string representation of that score.
   '''
   score = sentiment_score(sentence, analyser, translator)
   return score, sentiment(score)


def main():
    analyser = SentimentIntensityAnalyzer()
    translator = Translator()

    st.markdown('''
    # Sentiment analyzer

    [Link to GitHub](https://github.com).

    ## Analysis
    ''')

    sentence = st.text_area('Write your sentence')

    if st.button('Submit'):
        score, result = sentiment_analyzer_scores(sentence, analyser, translator)
        st.success(f'The sentiment of your text is **{result}** ({score :.2f}).')

if __name__ == '__main__':
    main()
