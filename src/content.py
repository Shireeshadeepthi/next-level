import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
import random
import os  # To handle file paths

# Step 1: Set Up the Environment
def setup_environment():
    # Load the model and tokenizer
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Replace with the actual model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

# Step 2: Preprocess Lesson Content
def split_and_tag(text, subject, grade, tokenizer, chunk_size=512):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    chunk_texts = [tokenizer.decode(chunk) for chunk in chunks]
    tagged_chunks = [
        {"subject": subject, "grade": grade, "content": chunk}
        for chunk in chunk_texts
    ]
    return tagged_chunks

# Step 3: Generate Topics
def extract_topics(chunk, model, tokenizer):
    prompt = f"Extract the most relevant topics from the following lesson content that are likely to appear in exams:\n\n{chunk}\n\nReturn the topics as a list."
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=1000 )
    topics = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return topics.split(", ")

# Step 4: Generate Questions
def generate_questions(topic, subject, model, tokenizer):
    prompt = f"Generate 5 exam-style questions for the topic '{topic}' in {subject} for CBSE Class 9. Ensure the questions are of medium difficulty and avoid duplication."
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=500)
    questions = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return questions.split("\n")

# Step 5: Create Question Papers
def create_question_paper(question_bank, num_questions=10):
    all_questions = [q for questions in question_bank.values() for q in questions]
    selected_questions = random.sample(all_questions, min(num_questions, len(all_questions)))
    return selected_questions

# Step 6: Validate Questions
def validate_question(question, model, tokenizer):
    prompt = f"Review the following question for CBSE Class 9 and check for profanity, subject relevance, duplication, and bias:\n\n{question}\n\nReturn 'Valid' or 'Invalid' with reasons."
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    validation_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return validation_result

# Main Function
def main():
    # Step 1: Set up the environment
    model, tokenizer = setup_environment()

    # Step 2: Define the folder structure
    base_folder = "class_9"  # Main folder
    subject_folder = "english"   # Subject folder (e.g., English)
    grade = "Class 9"

    # Step 3: Process each lesson file in the subject folder
    lesson_files = os.listdir(os.path.join(base_folder, subject_folder))
    for lesson_file in lesson_files:
        print(f"Processing {lesson_file}...")

        # Load lesson content
        with open(os.path.join(base_folder, subject_folder, lesson_file), "r") as file:
            lesson_content = file.read()

        # Split and tag lesson content
        chunks = split_and_tag(lesson_content, subject_folder, grade, tokenizer)

        # Save chunks to a JSON file
        lesson_name = os.path.splitext(lesson_file)[0]  # Remove .txt extension
        with open(f"{lesson_name}_chunks.json", "w") as file:
            json.dump(chunks, file)

        # Step 4: Generate topics
        all_topics = []
        for chunk in chunks:
            topics = extract_topics(chunk["content"], model, tokenizer)
            all_topics.extend(topics)
        all_topics = list(set(all_topics))

        # Save topics to a CSV file
        topics_df = pd.DataFrame(all_topics, columns=["Topic"])
        topics_df.to_csv(f"{lesson_name}_topics.csv", index=False)

        print("topics are saved")

        # Step 5: Generate questions
        question_bank = {}
        for topic in all_topics:
            questions = generate_questions(topic, subject_folder, model, tokenizer)
            question_bank[topic] = questions

        # Save questions to a JSON file
        with open(f"{lesson_name}_question_bank.json", "w") as file:
            json.dump(question_bank, file)

        # Step 6: Create question papers
        question_paper = create_question_paper(question_bank)
        with open(f"{lesson_name}_question_paper.json", "w") as file:
            json.dump(question_paper, file)

        # Step 7: Validate questions
        validated_questions = {}
        for topic, questions in question_bank.items():
            validated_questions[topic] = [validate_question(q, model, tokenizer) for q in questions]

        # Save validated questions to a JSON file
        with open(f"{lesson_name}_validated_questions.json", "w") as file:
            json.dump(validated_questions, file)

    print("Process completed successfully! Check the output files.")

# Run the script
if __name__ == "__main__":
    main()
