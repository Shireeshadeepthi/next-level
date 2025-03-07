import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
import random
import os
from torch.cuda.amp import autocast

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Set Up the Environment
def setup_environment():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)  # Move model to GPU
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
def extract_topics(chunks, model, tokenizer):
    prompts = [
        f"Extract the most relevant topics from the following lesson content that are likely to appear in exams:\n\n{chunk['content']}\n\nReturn the topics as a list."
        for chunk in chunks
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad(), autocast():  # Enable mixed precision
        outputs = model.generate(**inputs, max_length=200)

    topics = [tokenizer.decode(output, skip_special_tokens=True).split(", ") for output in outputs]
    return topics

# Step 4: Generate Questions
def generate_questions(topics, subject, model, tokenizer):
    prompts = [
        f"Generate 5 exam-style questions for the topic '{topic}' in {subject} for CBSE Class 9. Ensure the questions are of medium difficulty and avoid duplication."
        for topic in topics
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad(), autocast():  # Enable mixed precision
        outputs = model.generate(**inputs, max_length=200)

    questions = [tokenizer.decode(output, skip_special_tokens=True).split("\n") for output in outputs]
    return questions

# Step 5: Create Question Papers
def create_question_paper(question_bank, num_questions=10):
    all_questions = [q for questions in question_bank.values() for q in questions]
    selected_questions = random.sample(all_questions, min(num_questions, len(all_questions)))
    return selected_questions

# Step 6: Validate Questions
def validate_questions(questions, model, tokenizer):
    prompts = [
        f"Review the following question for CBSE Class 9 and check for profanity, subject relevance, duplication, and bias:\n\n{question}\n\nReturn 'Valid' or 'Invalid' with reasons."
        for question in questions
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad(), autocast():  # Enable mixed precision
        outputs = model.generate(**inputs, max_length=100)

    validation_results = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return validation_results

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
        topics = extract_topics(chunks, model, tokenizer)  # Process all chunks at once
        all_topics.extend([topic for sublist in topics for topic in sublist])
        all_topics = list(set(all_topics))

        # Save topics to a CSV file
        topics_df = pd.DataFrame(all_topics, columns=["Topic"])
        topics_df.to_csv(f"{lesson_name}_topics.csv", index=False)

        print("topics are saved")

        # Step 5: Generate questions
        question_bank = {}
        questions = generate_questions(all_topics, subject_folder, model, tokenizer)  # Process all topics at once
        for topic, question_list in zip(all_topics, questions):
            question_bank[topic] = question_list

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
            validation_results = validate_questions(questions, model, tokenizer)  # Process all questions at once
            validated_questions[topic] = validation_results

        # Save validated questions to a JSON file
        with open(f"{lesson_name}_validated_questions.json", "w") as file:
            json.dump(validated_questions, file)

    print("Process completed successfully! Check the output files.")

# Run the script
if __name__ == "__main__":
    main()