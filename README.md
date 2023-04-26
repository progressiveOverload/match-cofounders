# pre_workout_with_data

This project is an attempt to make a ML model that can match co-founders based on their skills and experiences using cosine similarities and showing it on a web app using streamlit.

Unfortunately model gives co-founders with similar skills and experiences rather than complementary skills. We can say this model isn't working to reach our goals.

You can try this model by running working_matcher_with_tester_4.py file with adding streamlit run then location of this folder.
working_matcher works too but don't have test data.



DATA
Project includes dataset from Y-Combinator startups, you can access here:
https://github.com/ali-ce/datasets/blob/master/Y-Combinator/Startups.csv

Data needed to be pre-processed for our project. This included creating new dataset which includes founder's data and combining other open source data like motivation and values. Those data is compiled from LinkedIn referrals/testimonials. (Thanks for the resource advice Oscar!)

Some profiles didn't included testimonials which made data gathering process variable and hard to conclude, so i passed with founders/co-founders that didn't have this information.



PIPELINE
Project pipeline includes:
Data gathering from LinkedIn and open sources.
Data preprocessing* (Adding founder's motivations and values to the dataset)
Vectorizing skills and experiences
Looking to co-founder's cosine similarities
Rank them in order



IMPROVEMENTS
Possible Future Improvements:
Creating a model that gives complimentary options rather than similar ones
Testing the model (Includes labelling and formatting)


