import argparse
import pandas as pd
from llama_cpp import Llama

if __name__ == '__main__':
    csv_path = "./dataset_logging.csv"

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        df = pd.DataFrame(
            columns=[
                "model", 
                "conference_source", 
                "paper_id", 
                "abstract",
                "filepath",
                "question",
                "answer"
            ]
        )
        df.to_csv(csv_path, index=False)

    model_path = "./llm/gemma-3-4b-it-q4_k_m.gguf"

    llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=str)
    parser.add_argument("-a", "--abstract", type=str)
    parser.add_argument("-p", "--path", type=str)

    args = parser.parse_args()

    conference_source = args.source
    abstract = args.abstract
    filepath = args.path

    while(True):
        user_input = input("YOU: ")

        if user_input == "/new" or user_input == "/n":
            print("Please provide new abstract")
            conference_source = input("Conference Source: ")
            abstract = input("Abstract: ")
            filepath = input("Paper path: ")

        prompt = f"""
        ABSTRACT:
        {abstract}

        ---------------------------------------------------------------------------
        INSTRUCTION: 
        Answer user question based on given abstract.

        ---------------------------------------------------------------------------
        USER QUESTION: 
        """

        response = llm.create_chat_completion(
            messages = [
                { "role": "system", "content": prompt },
                { "role": "user", "content": user_input }
            ]
        )
        output = response["choices"][0]["message"]["content"]
        print(f"BOT: {output}")

        id = 1 if df.empty else df["paper_id"].max() + 1

        df.loc[len(df)] = {
            "model": llm.metadata["general.name"],
            "conference_source": conference_source,
            "paper_id": id,
            "abstract": abstract,
            "filepath": filepath,
            "question": user_input,
            "answer": output
        }

        df.to_csv(csv_path, index=False)

