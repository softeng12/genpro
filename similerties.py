import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# nltk.download('punkt')
# nltk.download('stopwords')
def compare_documents(document1, document2):
    tokens = [document1, document2]
    vectorizer = TfidfVectorizer().fit_transform(tokens)
    vectors = vectorizer.toarray()
    similarity_score = cosine_similarity(vectors[0].reshape(1, -1), vectors[1].reshape(1, -1))
    similarity_percentage = similarity_score[0][0] * 100
    return similarity_percentage

document1 = "Pakistan, officially known as the Islamic Republic of Pakistan, is a vibrant and diverse country located in South Asia. With a rich history dating back thousands of years, Pakistan is home to a diverse cultural heritage that encompasses various ethnic groups, languages, and traditions. The country is known for its stunning landscapes, ranging from the majestic peaks of the Karakoram and Himalayan mountain ranges to the fertile plains of the Indus River. Pakistan is also renowned for its historical sites, such as the ancient city of Mohenjo-daro and the Mughal-era architectural wonders in Lahore and Islamabad. Despite facing numerous challenges, Pakistan's resilient and hospitable people continue to strive for progress and development, making it a fascinating and captivating destination."
document2 = "Pakistan, often known as the Islamic Republic of Pakistan, is a vibrantly diverse country in South Asia. Pakistan is a country with a deep, centuries-long history and a treasure trove of many ethnicities, languages, and cultural traditions. The country's beautiful scenery, from the towering peaks of the Himalayan and Karakoram mountain ranges to the verdant plains along the Indus River, are what make it so alluring. The historical wonders of Pakistan, such as Mohenjo-daro and the breathtaking Mughal buildings in Lahore and Islamabad, are also well-known. Despite facing several difficulties, Pakistan's tenacious and kind people persist in their quest for advancement and growth, making the country an alluring and captivating travel destination."

similarity_percentage = compare_documents(document1, document2)
print(f"The similarity between the two documents is {similarity_percentage:.2f}%.")










# from transformers import BertTokenizer, BertModel
# from sklearn.metrics.pairwise import cosine_similarity

# import torch

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
# model.eval()


# def compare_documents(document1, document2):
# # Tokenize the documents
#     tokens = tokenizer([document1, document2], padding=True, truncation=True, return_tensors='pt')

# # Get the BERT embeddings for the documents
#     with torch.no_grad():
#         outputs = model(**tokens)
#         embeddings = torch.mean(outputs.last_hidden_state, dim=1)

#     # Calculate the cosine similarity between document embeddings
#         similarity_score = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))

#     # Convert the similarity score to a percentage
#         similarity_percentage = similarity_score[0][0] * 100

#     return similarity_percentage


# document1 = "Pakistan, officially known as the Islamic Republic of Pakistan, is a vibrant and diverse country located in South Asia. With a rich history dating back thousands of years, Pakistan is home to a diverse cultural heritage that encompasses various ethnic groups, languages, and traditions. The country is known for its stunning landscapes, ranging from the majestic peaks of the Karakoram and Himalayan mountain ranges to the fertile plains of the Indus River. Pakistan is also renowned for its historical sites, such as the ancient city of Mohenjo-daro and the Mughal-era architectural wonders in Lahore and Islamabad. Despite facing numerous challenges, Pakistan's resilient and hospitable people continue to strive for progress and development, making it a fascinating and captivating destination."
# document2 = "Pakistan, often known as the Islamic Republic of Pakistan, is a vibrantly diverse country in South Asia. Pakistan is a country with a deep, centuries-long history and a treasure trove of many ethnicities, languages, and cultural traditions. The country's beautiful scenery, from the towering peaks of the Himalayan and Karakoram mountain ranges to the verdant plains along the Indus River, are what make it so alluring. The historical wonders of Pakistan, such as Mohenjo-daro and the breathtaking Mughal buildings in Lahore and Islamabad, are also well-known. Despite facing several difficulties, Pakistan's tenacious and kind people persist in their quest for advancement and growth, making the country an alluring and captivating travel destination."

# similarity_percentage = compare_documents(document1, document2)
# print(f"The similarity between the two documents is {similarity_percentage:.2f}%.")
