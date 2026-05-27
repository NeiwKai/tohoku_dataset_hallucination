import sys
import pandas as pd
from llama_cpp import Llama

if __name__ == '__main__':
    # Final output
    dataset_logging_csv_path = "./dataset_logging.csv"
    try:
        dataset_logging_df = pd.read_csv(dataset_logging_csv_path)
    except FileNotFoundError:
        dataset_logging_df = pd.DataFrame(
            columns=[
                "question_id",
                "model", 
                "conference_source", 
                "paper_id", 
                "abstract",
                "filepath",
                "question",
                "answer"
            ]
        )
        dataset_logging_df.to_csv(dataset_logging_csv_path, index=False)

    # Question list
    question_list_csv_path = "./question_list.csv"
    try:
        question_list_df = pd.read_csv(question_list_csv_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Conference paper
    conference_paper_csv_path = "./conference_paper.csv"
    try:
        conference_paper_df = pd.read_csv(conference_paper_csv_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Model initialize
    model_path = "./llm/gemma-3-4b-it-q4_k_m.gguf"
    llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)

    # New write count
    write_count = 0

    # Pipeline loop
    for index, question_list_row in question_list_df.iterrows():
        # Eliminate duplicate workload
        question_id = question_list_row["question_id"]
        if question_id in dataset_logging_df["question_id"].values:
            continue



        # Mapping with question_list.csv
        question_id = question_list_row["question_id"]
        paper_id = question_list_row["paper_id"]
        question = question_list_row["question"]

        print(f"Question #{question_id}")

        matched_row = conference_paper_df[
            conference_paper_df["paper_id"] == paper_id
        ].iloc[0]

        conference_source = matched_row["conference_source"]
        abstract = matched_row["abstract"]
        filepath = matched_row["filepath"]


        prompt = f"""
        ABSTRACT:
        {abstract}

        ---------------------------------------------------------------------------
        INSTRUCTION: 
        Answer user question based on given abstract.
        """

        response = llm.create_chat_completion(
            messages = [
                { "role": "system", "content": prompt },
                { "role": "user", "content": question }
            ]
        )
        answer = response["choices"][0]["message"]["content"]
        llm.reset()

        dataset_logging_df.loc[len(dataset_logging_df)] = {
            "question_id": question_id,
            "model": llm.metadata.get("general.name", model_path),
            "conference_source": conference_source,
            "paper_id": paper_id,
            "abstract": abstract,
            "filepath": filepath,
            "question": question,
            "answer": answer
        }

        dataset_logging_df.to_csv(dataset_logging_csv_path, index=False, quoting=1)
        write_count += 1

    print(f"Finish with {write_count} rows update!")
    sys.exit(0)

