# pre_workout_with_data

This project is an attempt to make a ML model that can match co-founders based on their skills and experiences using cosine similarities and showing it on a web app using streamlit.


Although ML was an interesting topic but mostly a black box for me, coding was mostly done by ChatGpt, i take my pride in bug fixes, getting data altogether and prompting.

Project includes dataset from Y-Combinator startups, you can access here:
https://github.com/ali-ce/datasets/blob/master/Y-Combinator/Startups.csv

Data needed to be pre-processed for our project. This included creating new dataset which includes founder's data and combining other open source data like motivation and values. Those data is compiled from LinkedIn referrals/testimonials. (Thanks for the resource advice Oscar!)

Some profiles didn't included testimonials which made data gathering process variable and hard to conclude, so i passed with founders/co-founders that didn't have this information.

Task was based on matching co-founders with each others on skills and experience are, i wanted to deliver more with other aspects of gathered data but i think i need some time to get better at time and stress management.

So let's explain how model works:

1- We got our data with founders




Project pipeline includes:
\nData gathering from LinkedIn and open sources.
\nData preprocessing* (Adding founder's motivations and values to the dataset)
\nVectorizing skills and experiences
\nLooking to co-founder's cosine similarities
\nRank them in order

Possible Future Improvements:
Testing the model (Includes labelling and formatting)
Improving the depth of model by adding values and their motivation to the equatiion

