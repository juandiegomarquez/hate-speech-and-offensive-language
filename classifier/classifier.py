import pandas as pd
import helpers

if __name__ == '__main__':
    print ("Loading data to classify...")

    #Tweets obtained here: https://github.com/sashaperigo/Trump-Tweets
    input_data = 'trump_tweets.csv'
    
    df = pd.read_csv(input_data, encoding='latin1')
    trump_tweets = [str(x) for x in df.Text]
    trump_predictions = helpers.get_tweets_predictions(trump_tweets)

    print("Predicted values:\n")
    for i,t in enumerate(trump_tweets):
        print(f"Tweet: {t}")
        print(f"Label: {helpers.class_to_name(trump_predictions[i])}\n")

    print ("Calculate accuracy on labeled data")
    df = pd.read_csv('../data/labeled_data.csv')
    tweets = df['tweet'].values

    actuals = df['class'].values
    predictions = helpers.get_tweets_predictions(tweets)

    accuracy = round(sum(predictions == actuals)/len(predictions), 2)
    print (f"accuracy: {accuracy}")

# %%