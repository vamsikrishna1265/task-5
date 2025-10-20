# task-5
This task classifies consumer complaints into categories like credit reporting, debt collection, consumer loans, and mortgage. Using machine learning, the text data is cleaned, transformed, and analyzed to predict complaint types, helping automate complaint handling and improve customer support efficiency.
outputs:
The output shows that the dataset with 1,50,000 complaints was successfully loaded and cleaned for model training. After processing, about 28,804 samples were used for training and 7,201 for testing. Three machine learning models were built — Logistic Regression (76.86% accuracy), Naive Bayes (73.60%), and Random Forest (78.29%). Among them, Random Forest performed the best, showing that it can classify complaints more accurately.

During testing, when a sample input like “The mortgage company applied a penalty by mistake” was entered, the model correctly predicted the category as Mortgage. The model was further tested on 20 different complaints, and it consistently predicted suitable categories such as Credit reporting, Debt collection, and Mortgage based on the text meaning. This result proves that the system can read and understand complaint texts, classify them correctly, and help automate customer service tasks with an accuracy close to 80%.
<img width="2879" height="1793" alt="image" src="https://github.com/user-attachments/assets/6b65626c-9a4e-41ff-be32-ad4b220a14ca" />
<img width="2879" height="1799" alt="image" src="https://github.com/user-attachments/assets/f3556dc0-2fff-4716-ab3a-9a4d05bb1704" />

