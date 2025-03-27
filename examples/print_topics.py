from deep_reason.topic_modeling import extract_topics_from_documents, visualize_topics
from deep_reason.utils import load_obliqa_dataset

chunks = load_obliqa_dataset(obliqa_dir="datasets/ObliQA/StructuredRegulatoryDocuments")
documents = [chunk.text for chunk in chunks]
# documents = [
#     "This is a document about artificial intelligence and machine learning.",
#     "Neural networks have revolutionized computer vision tasks.",
#     "Natural language processing enables computers to understand human language.",
#     "Climate change is affecting global weather patterns.",
#     "Renewable energy sources are becoming more affordable.",
#     "Deep learning architectures include convolutional and recurrent networks.",
#     "Transformer models have improved translation quality significantly.",
#     "Global warming has increased the frequency of extreme weather events.",
#     "Solar panels convert sunlight directly into electricity.",
#     "Wind turbines generate power from moving air."
# ]

# Extract topics with a smaller minimum topic size
print("Processing documents and extracting topics...")
result = extract_topics_from_documents(documents, min_topic_size=2)  # Reduced from default 10

# Print topic information from the model
print("\nDetailed Topic Information:")
print(result["topic_info"])

# Print topics and their representations
print("\nTopics extracted:")
for topic_id, words in result["topic_representations"].items():
    print(f"Topic {topic_id}: {words}")

# Print document-topic assignments
print("\nDocument-Topic Assignments:")
for doc_info in result["document_info"]:
    print(f"Document: {doc_info['document']}")
    print(f"Assigned Topic: {doc_info['topic']}")
    print(f"Probability: {doc_info['probability']:.4f}")
    print("---")

# Calculate topic distribution
topic_counts = {}
for topic in result["topics"]:
    if topic not in topic_counts:
        topic_counts[topic] = 0
    topic_counts[topic] += 1

print("\nTopic Distribution:")
for topic, count in topic_counts.items():
    if topic != -1:  # Skip outliers
        print(f"Topic {topic}: {count} documents ({count/len(documents)*100:.1f}%)")

# Organize documents by topic for a clearer view
docs_by_topic = {}
for i, topic in enumerate(result["topics"]):
    if topic not in docs_by_topic:
        docs_by_topic[topic] = []
    docs_by_topic[topic].append(documents[i])

print("\nDocuments by Topic:")
for topic_id, topic_docs in docs_by_topic.items():
    if topic_id != -1:  # Skip outliers
        print(f"Topic {topic_id} - {result['topic_representations'].get(topic_id, [('Unknown', 1)])[0][0]}:")
        for i, doc in enumerate(topic_docs, 1):
            truncated_doc = doc[:70] + "..." if len(doc) > 70 else doc
            print(f"  {i}. {truncated_doc}")
        print() 