import pandas as pd
import openai
import argparse
import pickle
from config import OPENAI_API_KEY
from model import DocumentRetriever, AnswerGenerator, FineTunedAnswerRetriever, HyDEGenerator, MultiqueryGenerator, ContextScorer

# Load models from files
with open('llm_embedder_model.pkl', 'rb') as f:
    llm_embedder_model = pickle.load(f)
with open('reranker.pkl', 'rb') as f:
    reranker = pickle.load(f)

#Prompt
default_prompt_fdaexpert = "Along with the question from the pharmaceutical industry, the regulatory guidelines most relevant to the question are given below in the form of contexts. Answer the question considering the given context. Focus on generating answers that prioritize more relevant and closely related regulatory guidelines."
question_example_fdaexpert = """
In accordance with M3(R2) Nonclinical Safety Studies for the Conduct of Human Clinical Trials and Marketing Authorization for Pharmaceuticals , When is assessment of reversibility considered to be appropriate and is it important to demonstrate full reversibility or is it sufficient to demonstrate the potential for full reversibility?
"""
answer_example_fdaexpert = """
Evaluation of the potential for reversibility of toxicity (i.e., return to the original or normal condition) should be provided when there is severe toxicity in a nonclinical study with potential adverse clinical impact. The evaluation can be based on a study of reversibility or on a scientific assessment. The scientific assessment of reversibility can include the extent and severity of the pathologic lesion, the regenerative capacity of the organ system showing the effect and knowledge of other drugs causing the effect. Thus, recovery arms or studies are not always critical to conclude whether an adverse effect is reversible. The demonstration of full reversibility is not considered essential. A trend towards reversibility (decrease in incidence or severity), and scientific assessment that this trend would eventually progress to full reversibility, are generally sufficient. If full reversibility is not anticipated, this should be considered in the clinical risk assessment. A toxicity study that includes a terminal non-dosing period is generally warranted if a scientific assessment cannot predict whether the toxicity will be reversible and if:\n* there is severe toxicity at clinically relevant exposures (e.g., \\(\\leq 10\\)-fold the clinical exposure); or\n* the toxicity is only detectable at an advanced stage of the pathophysiology in humans and where significant reduction in organ function is expected. (The assessment of reversibility in this case should be considered even at \\(>10\\)-fold exposure multiples.) A toxicity study that includes a terminal non-dosing period is generally not warranted when the toxicity:\n* can be readily monitored in humans at an early stage before the toxicity becomes severe; or\n* is known to be irrelevant to humans (e.g., rodent Harderian gland toxicity); or\n* is only observed at high exposures not considered clinically relevant (see 2 above for exception); or\n* is similar to that induced by related agents, and the toxicity based on prior clinical experience with these related agents is considered a manageable risk.\nIf a study of reversibility is called for, it should be available to support clinical studies of a duration similar to those at which the adverse effects were seen nonclinically. However, a reversibility study is generally not warranted to support clinical trials of a duration equivalent to that at which the adverse effect was not observed nonclinically. If a particular lesion is demonstrated to be reversible in a short duration (e.g., 2-week or 1-month) study, and does not progress in severity in longer term studies, repeating the reversibility assessment in longer term toxicity studies is generally not warranted. If a reversibility study is warranted, it is efficient to conduct it as part of a chronic study so that all toxicities of concern can be assessed in a single study, provided that it is not critical to conduct it earlier to support a specific clinical trial.
 """


# Initialize OpenAI client
client = openai.Client(api_key=OPENAI_API_KEY)

#Dataframe to remove the Q&A document which contains the exact answer of the given test question
#This part was added for the test which was conducted in the paper
df = pd.read_csv('Data/QnA.csv')

# Initialize Document Retriever
persist_directory = 'Data/DB_FAISS'
document_retriever = DocumentRetriever(df, llm_embedder_model, 'reranker', client, persist_directory)

# Initialize other components
fine_tuned_answer_retriever = FineTunedAnswerRetriever(client)
hyde_retriever = HyDEGenerator(client)
multiquery_generator = MultiqueryGenerator(client, unique_num = 2) #unique num means the number of additional queries to generate

# Initialize ContextScorer
context_scorer = ContextScorer(
    document_retriever=document_retriever, 
    client=client,
    model=reranker.model,
    tokenizer=reranker.tokenizer
)

# Initialize Answer Generator
answer_generator = AnswerGenerator(
    context_scorer=context_scorer,
    fine_tuned_answer_retriever=fine_tuned_answer_retriever,
    hyde_retriever=hyde_retriever,
    client=client,
    multiquery_generator=multiquery_generator,
    document_retriever=document_retriever,
    scoring_method='reranker',  # or 'reranker' based on your choice
    answer_model="gpt-3.5"  # adjust as needed
)

def QA_RAG(question, answer_generator, top_k, top_k_ans, num_docs, num_docs_ans, final_k, num_repeats=1, model = reranker.model, tokenizer = reranker.tokenizer):
    unique_top_contexts, _ = answer_generator.get_combined_top_contexts_with_ft(
        question, num_repeats=num_repeats, top_k=top_k, top_k_ans=top_k_ans, num_docs=num_docs, num_docs_ans = num_docs_ans, final_k=final_k, model = model, tokenizer = tokenizer)
    f_template = answer_generator.generate_final_template_without_score(question = question, combined_top_contexts = unique_top_contexts, default_prompt = default_prompt_fdaexpert)
    retrieved_answer = answer_generator.get_answer_without_score(client = client, question_example = question_example_fdaexpert, answer_example = answer_example_fdaexpert, final_template = f_template)
    return retrieved_answer

def Multiquery(question, answer_generator, top_k, top_k_ans, num_docs, num_docs_ans, final_k, unique_num = 2, num_repeats=1, model = reranker.model, tokenizer = reranker.tokenizer):
    unique_top_contexts, _ = answer_generator.get_combined_top_contexts_with_ft_multiquery(
        question, num_repeats=num_repeats, top_k=top_k, top_k_ans=top_k_ans, num_docs=num_docs, num_docs_ans = num_docs_ans, unique_num = unique_num, final_k = final_k, model = model, tokenizer = tokenizer)
    f_template = answer_generator.generate_final_template_without_score(question = question, combined_top_contexts = unique_top_contexts, default_prompt = default_prompt_fdaexpert)
    retrieved_answer = answer_generator.get_answer_without_score(client=client, question_example = question_example_fdaexpert, answer_example = answer_example_fdaexpert, final_template = f_template)
    return retrieved_answer

def HyDE(question, answer_generator, top_k, top_k_ans, num_docs, num_docs_ans, final_k, num_repeats=1, model = reranker.model, tokenizer = reranker.tokenizer):
    unique_top_contexts, _ = answer_generator.get_combined_top_contexts_with_HyDE(
        question, num_repeats=num_repeats, top_k=top_k, top_k_ans=top_k_ans, num_docs=num_docs, num_docs_ans = num_docs_ans, final_k=final_k, model = model, tokenizer = tokenizer)
    f_template = answer_generator.generate_final_template_without_score(question = question, combined_top_contexts = unique_top_contexts, default_prompt = default_prompt_fdaexpert)
    retrieved_answer = answer_generator.get_answer_without_score(client = client, question_example = question_example_fdaexpert, answer_example = answer_example_fdaexpert, final_template = f_template)
    return retrieved_answer

if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Parameters for the execution of QA-RAG")

    # Add arguments
    parser.add_argument('--question', type=str, required=True, help='Enter the question')
    parser.add_argument('--top_k', type=int, default=6, help='Number of top documents to retrieve')
    parser.add_argument('--top_k_ans', type=int, default=6, help='Number of top documents to retrieve from the answer')
    parser.add_argument('--num_docs', type=int, default=12, help='Number of documents for initial retrieval')
    parser.add_argument('--num_docs_ans', type=int, default=12, help='Number of documents for initial retrieval from the answer')
    parser.add_argument('--final_k', type=int, default=6, help='Number of final top documents to return after reranking')
    parser.add_argument('--num_repeats', type=int, default=1, help='Number of repeats for answer retrieval')
    parser.add_argument('--method', type=str, choices=['QA_RAG', 'Multiquery', 'HyDE'], required=True, help='Choose the method to use: QA_RAG, Multiquery, or HyDE')

    # Parse the arguments
    args = parser.parse_args()


    if args.method == 'QA_RAG':
        retrieved_answer = QA_RAG(
        question=args.question,
        answer_generator=answer_generator,
        top_k=args.top_k,
        top_k_ans=args.top_k_ans,
        num_docs=args.num_docs,
        num_docs_ans=args.num_docs_ans,
        final_k=args.final_k,
        num_repeats=args.num_repeats
        )
    elif args.method == 'Multiquery':
        retrieved_answer = Multiquery(
        question=args.question,
        answer_generator=answer_generator,
        top_k=args.top_k,
        top_k_ans=args.top_k_ans,
        num_docs=args.num_docs,
        num_docs_ans=args.num_docs_ans,
        final_k=args.final_k,
        num_repeats=args.num_repeats    
        )
    elif args.method == 'HyDE':
        retrieved_answer = HyDE(
        question=args.question,
        answer_generator=answer_generator,
        top_k=args.top_k,
        top_k_ans=args.top_k_ans,
        num_docs=args.num_docs,
        num_docs_ans=args.num_docs_ans,
        final_k=args.final_k,
        num_repeats=args.num_repeats
        )

    # Print the retrieved answer
    print(retrieved_answer)
