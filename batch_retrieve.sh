# Let's do a batch retrieval
# First let's do the not finetuned
python main.py -q ./data/topics_1.json -d ./data/Answers.json -be sentence-transformers/all-MiniLM-L6-v2 -ce cross-encoder/ms-marco-MiniLM-L-6-v2
python main.py -r -q ./data/topics_1.json -d ./data/Answers.json -be sentence-transformers/all-MiniLM-L6-v2 -ce cross-encoder/ms-marco-MiniLM-L-6-v2
python main.py -q ./data/topics_2.json -d ./data/Answers.json -be sentence-transformers/all-MiniLM-L6-v2 -ce cross-encoder/ms-marco-MiniLM-L-6-v2
python main.py -r -q ./data/topics_2.json -d ./data/Answers.json -be sentence-transformers/all-MiniLM-L6-v2 -ce cross-encoder/ms-marco-MiniLM-L-6-v2

# And finetuned after that
python main.py -q ./data/topics_1.json -d ./data/Answers.json -be sentence-transformers/all-MiniLM-L6-v2 -ce cross-encoder/ms-marco-MiniLM-L-6-v2 -ft
python main.py -r -q ./data/topics_1.json -d ./data/Answers.json -be sentence-transformers/all-MiniLM-L6-v2 -ce cross-encoder/ms-marco-MiniLM-L-6-v2 -ft
python main.py -q ./data/topics_2.json -d ./data/Answers.json -be sentence-transformers/all-MiniLM-L6-v2 -ce cross-encoder/ms-marco-MiniLM-L-6-v2 -ft
python main.py -r -q ./data/topics_2.json -d ./data/Answers.json -be sentence-transformers/all-MiniLM-L6-v2 -ce cross-encoder/ms-marco-MiniLM-L-6-v2 -ft
