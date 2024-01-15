from langchain_community.vectorstores import FAISS
import re
import torch
from parsers import LineListOutputParser


#Prompt
system_message = "Along with the question from the pharmaceutical industry, the regulatory guidelines most relevant to the question are given below in the form of contexts. Answer the question considering the given context. Focus on generating answers that prioritize more relevant and closely related regulatory guidelines."
EX_QUESTION = """
In accordance with M3(R2) Nonclinical Safety Studies for the Conduct of Human Clinical Trials and Marketing Authorization for Pharmaceuticals , When is assessment of reversibility considered to be appropriate and is it important to demonstrate full reversibility or is it sufficient to demonstrate the potential for full reversibility?
"""
EX_ANSWER = """
Evaluation of the potential for reversibility of toxicity (i.e., return to the original or normal condition) should be provided when there is severe toxicity in a nonclinical study with potential adverse clinical impact. The evaluation can be based on a study of reversibility or on a scientific assessment. The scientific assessment of reversibility can include the extent and severity of the pathologic lesion, the regenerative capacity of the organ system showing the effect and knowledge of other drugs causing the effect. Thus, recovery arms or studies are not always critical to conclude whether an adverse effect is reversible. The demonstration of full reversibility is not considered essential. A trend towards reversibility (decrease in incidence or severity), and scientific assessment that this trend would eventually progress to full reversibility, are generally sufficient. If full reversibility is not anticipated, this should be considered in the clinical risk assessment. A toxicity study that includes a terminal non-dosing period is generally warranted if a scientific assessment cannot predict whether the toxicity will be reversible and if:\n* there is severe toxicity at clinically relevant exposures (e.g., \\(\\leq 10\\)-fold the clinical exposure); or\n* the toxicity is only detectable at an advanced stage of the pathophysiology in humans and where significant reduction in organ function is expected. (The assessment of reversibility in this case should be considered even at \\(>10\\)-fold exposure multiples.) A toxicity study that includes a terminal non-dosing period is generally not warranted when the toxicity:\n* can be readily monitored in humans at an early stage before the toxicity becomes severe; or\n* is known to be irrelevant to humans (e.g., rodent Harderian gland toxicity); or\n* is only observed at high exposures not considered clinically relevant (see 2 above for exception); or\n* is similar to that induced by related agents, and the toxicity based on prior clinical experience with these related agents is considered a manageable risk.\nIf a study of reversibility is called for, it should be available to support clinical studies of a duration similar to those at which the adverse effects were seen nonclinically. However, a reversibility study is generally not warranted to support clinical trials of a duration equivalent to that at which the adverse effect was not observed nonclinically. If a particular lesion is demonstrated to be reversible in a short duration (e.g., 2-week or 1-month) study, and does not progress in severity in longer term studies, repeating the reversibility assessment in longer term toxicity studies is generally not warranted. If a reversibility study is warranted, it is efficient to conduct it as part of a chronic study so that all toxicities of concern can be assessed in a single study, provided that it is not critical to conduct it earlier to support a specific clinical trial.
 """


def modify_document(rel_doc):
    """
    Function to the modify documents used in the test done in the paper.
    """
    # Remove month_year form at the end of the 'rel_doc'
    modified_rel_doc = re.sub(r'_(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)_\d{4}', '', rel_doc)

    modified_rel_doc = modified_rel_doc.replace('_', ' ')

    return modified_rel_doc


#Class to retrieve relevant documents
class DocumentRetriever:
    def __init__(self, df, llm_embedder_model, scoring_method, client, persist_directory):
        self.df = df
        self.llm_embedder_model = llm_embedder_model
        self.scoring_method = scoring_method
        self.client = client
        self.persist_directory = persist_directory
        self.context_cache = {} 


    def get_context(self, question, num_docs):
        """
        This function finds the most relevant contexts for a given question.
        :param question: String, the user's question.
        :param num_docs: Number of documents to search.
        :return: List, list of relevant documents.
        """
        if question in self.context_cache:
            return self.context_cache[question]

        print(f"Processing question: {question}")

        #Code to remove the Q&A document which contains the exact answer of the given test question
        #This part was added for the test which was conducted in the paper
        if '\n  Relevant text, if any:' in question:
          q_temp = question.split('\n  Relevant text, if any:')[0]
        else:
          q_temp = question

        if not self.df[self.df.Question == q_temp].empty:
            rel_doc = self.df[self.df.Question == q_temp]['Title'].iloc[0]
            qa = self.df[self.df.Question == q_temp]['Question'].iloc[0]
        else:
            qa = question

        vectordb = FAISS.load_local(self.persist_directory, embeddings = self.llm_embedder_model)

        if 'rel_doc' in locals():
            target_word = modify_document(rel_doc)
            faiss_dict = vectordb.docstore._dict
            document_vals = faiss_dict.values()
            matching_keys = []

            for key_val in document_vals:
                modified_source = key_val.metadata['source'].replace('-', ' ')
                if target_word in modified_source:
                    matching_keys += [key for key, value in faiss_dict.items() if target_word in value.metadata['source'].replace('-', ' ')]
                    matching_keys = list(set(matching_keys))
            if len(matching_keys) != 0:
                vectordb.delete(ids=matching_keys)
            relevant_documents = vectordb.similarity_search(qa, k = num_docs)

        else:
            relevant_documents = vectordb.similarity_search(qa, k = num_docs)

        self.context_cache[question] = relevant_documents

        return relevant_documents


    def get_context_multiquery(self, questions, num_docs):
        """
        This function finds contexts for multiple questions.
        :param questions: List, several questions.
        :param num_docs: Number of documents to search.
        :return: Dictionary, list of relevant documents for each question.
        """

        original_q = questions[0]

        if original_q in self.context_cache:
            return self.context_cache[original_q]

        print(f"Processing question: {original_q}")

        #Code to remove the Q&A document which contains the exact answer of the given test question
        #This part was added for the test which was conducted in the paper
        if '\n  Relevant text, if any:' in original_q:
          q_temp = original_q.split('\n  Relevant text, if any:')[0]
        else:
          q_temp = original_q

        if not self.df[self.df.Question == q_temp].empty:
            rel_doc = self.df[self.df.Question == q_temp]['Title'].iloc[0]
            qa = self.df[self.df.Question == q_temp]['Question'].iloc[0]
        else:
            qa = q_temp

        vectordb = FAISS.load_local(self.persist_directory, embeddings = self.llm_embedder_model)

        relevant_documents = []

        if 'rel_doc' in locals():
            target_word = modify_document(rel_doc)
            faiss_dict = vectordb.docstore._dict
            document_vals = faiss_dict.values()
            matching_keys = []

            for key_val in document_vals:
                modified_source = key_val.metadata['source'].replace('-', ' ') 
                if target_word in modified_source:
                    matching_keys += [key for key, value in faiss_dict.items() if target_word in value.metadata['source'].replace('-', ' ')]
                    matching_keys = list(set(matching_keys))

            if len(matching_keys) != 0:
                vectordb.delete(ids=matching_keys)
        for question in questions:
            relevant_documents += vectordb.similarity_search(question, k=num_docs)
        self.context_cache[original_q] = relevant_documents
        return relevant_documents


    def get_template(self, content, num_docs, question):
        # Function to get template
        c = self.get_context(content, num_docs)

        results = []
        for context in c:
            content = context.page_content
            if '/content/drive/MyDrive/Dataset/FDA_guideline_OCR2/' in context.metadata['source']:
                source = context.metadata['source'].split('/content/drive/MyDrive/Dataset/FDA_guideline_OCR2/')[1].rstrip('.mmd')
            elif '/content/drive/MyDrive/Dataset/ICH_guideline_OCR/' in context.metadata['source']:
                source = context.metadata['source'].split('/content/drive/MyDrive/Dataset/ICH_guideline_OCR/')[1].rstrip('.mmd')
            else:
                source = context.metadata['source'].strip('.mmd')

            template = f"""
--------------------
Question: {question}
--------------------
Context: {content}
--------------------
Context title: {source}
"""
            results.append(template)

        return results


    def get_full_context_by_index(self, context, score, question):
        # Function to get full context by index
        content = context.page_content
        if '/content/drive/MyDrive/Dataset/FDA_guideline_OCR2/' in context.metadata['source']:
            source = context.metadata['source'].split('/content/drive/MyDrive/Dataset/FDA_guideline_OCR2/')[1].rstrip('.mmd')
        elif '/content/drive/MyDrive/Dataset/ICH_guideline_OCR/' in context.metadata['source']:
            source = context.metadata['source'].split('/content/drive/MyDrive/Dataset/ICH_guideline_OCR/')[1].rstrip('.mmd')
        else:
            source = context.metadata['source'].strip('.mmd')

        template = f"""
--------------------
Question: {question}
--------------------
Context: {content}
--------------------
Context title: {source}
"""

        if self.scoring_method == 'scoring_agent':
          context_detail = f"{template}--------------------\nRelevance with the question: {score}/10"
        elif self.scoring_method == 'reranker':
          context_detail = f"{template}--------------------\nRelevance with the question: {score}"

        return context_detail


    def get_full_context_by_index_multiquery(self, context, score, questions):
        question = questions[0]

        content = context.page_content
        if '/content/drive/MyDrive/Dataset/FDA_guideline_OCR2/' in context.metadata['source']:
            source = context.metadata['source'].split('/content/drive/MyDrive/Dataset/FDA_guideline_OCR2/')[1].rstrip('.mmd')
        elif '/content/drive/MyDrive/Dataset/ICH_guideline_OCR/' in context.metadata['source']:
            source = context.metadata['source'].split('/content/drive/MyDrive/Dataset/ICH_guideline_OCR/')[1].rstrip('.mmd')
        else:
            source = context.metadata['source'].strip('.mmd')

        template = f"""
--------------------
Question: {question}
--------------------
Context: {content}
--------------------
Context title: {source}
"""

        if self.scoring_method == 'scoring_agent':
          context_detail = f"{template}--------------------\nRelevance with the question: {score}/10"
        elif self.scoring_method == 'reranker':
          context_detail = f"{template}--------------------\nRelevance with the question: {score}"

        return context_detail



#Class to generate Multiquery
class MultiqueryGenerator:
    def __init__(self, client, unique_num):
        self.client = client
        self.output_parser = LineListOutputParser()
        self.unique_num = unique_num

    def generate_additional_questions(self, original_question, unique_num):
        # Function to generate additional questions
        sys_mess = f"""You are an AI language model assistant. Your task is
to generate {unique_num} different versions of the given user
question to retrieve relevant documents from a vector database.
By generating multiple perspectives on the user question,
your goal is to help the user overcome some of the limitations
of distance-based similarity search. Provide these alternative
questions separated by newlines."""

        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": sys_mess},
                {"role": "user", "content": original_question}
            ]
        )

        # Parsing the model's response using LineListOutputParser
        response_text = response.choices[0].message.content
        line_list = self.output_parser.parse(response_text)

        # Returns modified questions to the list, but removes empty strings
        modified_questions = [q for q in line_list.lines if q.strip() != '']
        return [original_question] + modified_questions



class HyDEGenerator:
    def __init__(self, client):
        self.client = client

    def generate_hyde(self, question):
        """
        Generates a hypothetical document (HyDE) to answer a given question.
        :param question: String, the user's question.
        :return: String, the generated hypothetical document.
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": "You are an expert on FDA guidelines, providing answers and clarity on various topics related to the agency's regulations and recommendations."},
                {"role": "user", "content": f"""
                Please write a passage to answer the question
Question: {question}
Passage:
"""
                }
            ]
        )
        answer = response.choices[0].message.content
        return answer



#Class to generate answers from fine-tuned LLM
class FineTunedAnswerRetriever:
    def __init__(self, client):
        self.client = client

    def return_finetuned_answer(self, question):
        """
        Returns an answer derived from fine-tuned LLM for a given question.
        :param question: String, the user's question.
        :return: String, the answer from fine-tuned LLM.
        """
        response = self.client.chat.completions.create(
            model="ft:gpt-3.5-turbo-1106:personal::8SeL1ADw",
            messages=[
                {"role": "system", "content": "You are an expert on FDA guidelines, providing answers and clarity on various topics related to the agency's regulations and recommendations."},
                {"role": "user", "content": f"{question}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer



#Class to score the context with the query
class ContextScorer:
    def __init__(self, document_retriever, client, model=None, tokenizer=None):
        """
        Constructor for the ContextScorer class.
        Initializes the class with a document retriever, a client for API interactions, and optionally a model and tokenizer for scoring.
        :param document_retriever: Instance of DocumentRetriever for retrieving relevant documents.
        :param client: Client for making API calls.
        :param model: Reranking Model used for scoring contexts.
        :param tokenizer: Tokenizer used in conjunction with the model.
        """
        self.document_retriever = document_retriever
        self.client = client
        self.model = model
        self.tokenizer = tokenizer


    def calculate_top_contexts_for_single_question(self, question, top_k, num_docs):
        """
        Calculates and returns the top contexts for a single question using the scoring agent.
        :param question: String, the user's question.
        :param top_k: Integer, number of top contexts to retrieve.
        :param num_docs: Integer, number of documents to consider for scoring.
        :return: List of top contexts details based on the scoring method.
        """
        system_message_for_question = """
        You are a helpful AI assistant.
        Solve tasks using your language skills.
        The “Question” below is from someone in the pharmaceutical industry, and the “Context” is a guidance document that I think is relevant to answer this question. “Context title" is the title of the document where the context has been extracted. Based on the content of the “Context” and the “Context name”, Please score the extent to which this context helps you solve the question. On a scale of 1 to 10.
        Be sure to only return a score (number) and do not add any characters or even a space. It's very important!
        """
        contexts = self.document_retriever.get_context(question, num_docs)
        scores = self.calculate_scores_agent(contexts, self.client, system_message_for_question, question)
        return self.get_top_contexts_details(contexts, scores, top_k, question)


    def calculate_top_contexts_for_single_answer(self, answer, question, top_k_ans, num_docs):
        """
        Calculates and returns the top contexts for a single answer using the scoring agent.
        :param answer: String, the answer to the question.
        :param question: String, the user's question.
        :param top_k_ans: Integer, number of top contexts to retrieve for answers.
        :param num_docs: Integer, number of documents to consider for scoring answers.
        :return: List of top contexts details based on the scoring method.
        """
        system_message_for_answer = """
        You are a helpful AI assistant.
        Solve tasks using your language skills.
        The "Answer" below is an answer for a question from someone in the pharmaceutical industry, and the “Context” is a guidance document that I think is relevant to answer this answer. “Context title" is the title of the document where the context has been extracted. Based on the content of the “Context” and the “Context name”, Please score the extent to which this context helps you solve the question. On a scale of 1 to 10.
        Be sure to only return a score (number) and do not add any characters or even a space. It's very important!
        """
        contexts = self.document_retriever.get_context(answer, num_docs)
        scores = self.calculate_scores_agent(contexts, self.client, system_message_for_answer, answer)
        return self.get_top_contexts_details(contexts, scores, top_k_ans, question)


    def calculate_scores_agent(self, templates, client, system_message, input_text):
        """
        Calculates scores using a scoring agent.
        :param templates: List of templates containing context details.
        :param client: Client for API calls.
        :param system_message: String, system message for context.
        :param input_text: String, input text for scoring.
        :return: List of scores for each context.
        """
        scores = []
        for template in templates:
            successful = False
            while not successful:
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo-1106",
                        messages=[
                            {"role": "system", "content": f"{system_message}"},
                            {"role": "user", "content": f"{EX_QUESTION}"},
                            {"role": "assistant", "content": f"{EX_ANSWER}"},
                            {"role": "user", "content": f"{template}"}
                        ]
                    )
                    score = float(response.choices[0].message.content)
                    scores.append(score)
                    successful = True
                    print("Scoring success")

                except Exception:
                    print("Error in response format, retrying...")
        return scores


    def calculate_reranked_score_for_single_question(self, question, model, tokenizer, top_k, num_docs):
        """
        Calculates and returns the top contexts for a single question using the reranker.
        :param question: String, the user's question.
        :param model: Reranker model used for scoring contexts.
        :param tokenizer: Tokenizer used in conjunction with the model.
        :param top_k: Integer, number of top contexts to retrieve.
        :param num_docs: Integer, number of documents to consider for scoring.
        :return: List of top contexts details based on the reranker.
        """
        contexts = self.document_retriever.get_context(question, num_docs)
        scores = self.score_pairs(question, contexts, model, tokenizer)
        return self.get_top_contexts_details(contexts = contexts, scores = scores, top_k = top_k, question = question)

    def calculate_reranked_score_for_single_question_multiquery(self, questions, model, tokenizer, top_k, num_docs):
        """
        Calculates and returns the top contexts for multiquery questions using the reranker.
        :param questions: List, multiple questions.
        :param model: Reranker model used for scoring contexts.
        :param tokenizer: Tokenizer used in conjunction with the model.
        :param top_k: Integer, number of top contexts to retrieve.
        :param num_docs: Integer, number of documents to consider for scoring.
        :return: List of top contexts details based on the reranker.
        """
        contexts = self.document_retriever.get_context_multiquery(questions, num_docs)
        scores = self.score_pairs_multiquery(questions, contexts, model, tokenizer)
        return self.get_top_contexts_details_multiquery(contexts = contexts, scores = scores, top_k = top_k, questions = questions)

    def calculate_scoringagent_score_for_single_question_multiquery(self, questions, top_k, num_docs):
        """
        Calculates and returns the top contexts for multiquery questions using the scoring agent.
        :param questions: String, the multiquery questions.
        :param top_k: Integer, number of top contexts to retrieve.
        :param num_docs: Integer, number of documents to consider for scoring.
        :return: List of top contexts details based on the scoring method.
        """
        system_message_for_question = """
        You are a helpful AI assistant.
        Solve tasks using your language skills.
        The “Question” below is from someone in the pharmaceutical industry, and the “Context” is a guidance document that I think is relevant to answer this question. “Context title" is the title of the document where the context has been extracted. Based on the content of the “Context” and the “Context name”, Please score the extent to which this context helps you solve the question. On a scale of 1 to 10.
        Be sure to only return a score (number) and do not add any characters or even a space. It's very important!
        """
        contexts = self.document_retriever.get_context_multiquery(questions, num_docs)
        scores = self.scoringagent_score_pairs_multiquery(contexts, self.client, system_message_for_question, questions)

        return self.get_top_contexts_details_multiquery(contexts = contexts, scores = scores, top_k = top_k, questions = questions)


    def calculate_reranked_score_for_single_answer(self, answer, question, model, tokenizer, top_k_ans, num_docs):
        """
        Calculates and returns the top contexts for a single answer using the reranker.
        :param answer: String, the answer to the question.
        :param question: String, the user's question.
        :param model: Reranker model used for scoring contexts.
        :param tokenizer: Tokenizer used in conjunction with the model.
        :param top_k_ans: Integer, number of top contexts to retrieve for answers.
        :param num_docs: Integer, number of documents to consider for scoring answers.
        :return: List of top contexts details based on the reranker.
        """
        contexts = self.document_retriever.get_context(answer, num_docs)
        scores = self.score_pairs(question, contexts, model, tokenizer)
        return self.get_top_contexts_details(contexts = contexts, scores = scores, top_k = top_k_ans, question = question)

    def score_pairs(self, question, contexts, model, tokenizer):
        """
        Scores each pair of question and context using the provided reranker model.
        :param question: String, the user's question.
        :param contexts: List of context objects, each containing page content to be scored.
        :param model: The reranker model used for scoring.
        :param tokenizer: The tokenizer for preparing inputs for the model.
        :return: List of the result scores.
        """
        pairs = [[question, context.page_content] for context in contexts]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
            return scores.tolist()

    def score_pairs_multiquery(self, questions, contexts, model, tokenizer):
        """
        Scores each pair of multiquery questions and contexts using the reranker model.
        :param questions: List of strings, the user's questions.
        :param contexts: List of context objects, each containing page content to be scored.
        :param model: The reranker model used for scoring.
        :param tokenizer: The tokenizer for preparing inputs for the model.
        :return: List of the result scores.
        """
        question = questions[0]
        pairs = [[question, context.page_content] for context in contexts]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
            return scores.tolist()


    def scoringagent_score_pairs_multiquery(self, templates, client, system_message, input_text):
        """
        Scores each pair of multiquery questions and contexts using the scoring agent.
        :param templates: List of templates containing context details.
        :param client: Client for API calls.
        :param system_message: String, system message for context.
        :param input_text: String, input text for scoring.
        :return: List of scores for each context.
        """
        scores = []
        for template in templates:
            successful = False
            while not successful:
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo-1106",
                        messages=[
                            {"role": "system", "content": f"{system_message}"},
                            {"role": "user", "content": f"{EX_QUESTION}"},
                            {"role": "assistant", "content": f"{EX_ANSWER}"},
                            {"role": "user", "content": f"{template}"}
                        ]
                    )
                    score = float(response.choices[0].message.content)
                    scores.append(score)
                    successful = True
                    print("Scoring success")

                except Exception:
                    print("Error in response format, retrying...")

        return scores


    def get_top_contexts_details(self, contexts, scores, top_k, question):
        """
        Retrieves the top context details based on given scores.
        :param contexts: List of context objects.
        :param scores: List of scores corresponding to each context.
        :param top_k: Integer, number of top contexts to return.
        :param question: String, the user's question.
        :return: List of top context details.
        """
        scored_contexts = sorted(zip(scores, contexts), key=lambda x: x[0], reverse=True)
        top_contexts = scored_contexts[:top_k]
        top_contexts_details = []
        for score, context in top_contexts:
            context_detail = self.document_retriever.get_full_context_by_index(context, score, question)
            top_contexts_details.append(context_detail)
        return top_contexts_details

    def get_top_contexts_details_multiquery(self, contexts, scores, top_k, questions):
        """
        Retrieves the top context details for multiquery scenarios based on given scores.
        :param contexts: List of context objects.
        :param scores: List of scores corresponding to each context.
        :param top_k: Integer, number of top contexts to return.
        :param questions: List of strings, the multiquery questions.
        :return: List of top context details for each question.
        """
        scored_contexts = sorted(zip(scores, contexts), key=lambda x: x[0], reverse=True)
        top_contexts = scored_contexts[:top_k]
        top_contexts_details = []
        for score, context in top_contexts:
            context_detail = self.document_retriever.get_full_context_by_index_multiquery(context, score, questions)
            top_contexts_details.append(context_detail)
        return top_contexts_details



#Class to drop the duplicated contexts
class UniqueContextGenerator:
    def __init__(self, contexts):
        self.contexts = contexts
        self.relevance_phrase = '\n--------------------\nRelevance with the question:'

    def _split_contexts(self):
        """
        Splits contexts into main content and the relevance score parts.
        :return: Tuple containing lists of cleaned contexts and their relevance parts.
        """
        cleaned_contexts = []
        relevance_parts = []
        for context in self.contexts:
            main_content, relevance = context.rsplit(self.relevance_phrase, 1)
            cleaned_contexts.append(main_content)
            relevance_parts.append(relevance)
        return cleaned_contexts, relevance_parts

    def get_unique_contexts(self):
        """
        Filters out unique contexts from a list of contexts, removing duplicates.
        :return: A tuple of lists containing unique contexts and removed duplicate contexts.
        """
        cleaned_contexts, relevance_parts = self._split_contexts()

        # Removing duplicates
        unique_contents = list(dict.fromkeys(cleaned_contexts))
        unique_top_contexts = []
        removed_contexts = []
        content_counts = {content: cleaned_contexts.count(content) for content in cleaned_contexts}

        for i, context in enumerate(cleaned_contexts):
            full_context = context + self.relevance_phrase + relevance_parts[i]
            if content_counts[context] > 1:
                removed_contexts.append(full_context)
                content_counts[context] -= 1
            elif context in unique_contents:
                unique_top_contexts.append(full_context)

        return unique_top_contexts, removed_contexts



#Class for final answer generation
class AnswerGenerator:
    def __init__(self, context_scorer, fine_tuned_answer_retriever, hyde_retriever, client, multiquery_generator, document_retriever, scoring_method='scoring_agent', answer_model = "gpt-3.5"):
        """
        Constructor for the AnswerGenerator class.
        Initializes the class with various components for generating answers based on context and relevance score.
        :param context_scorer: An instance of ContextScorer for scoring contexts.
        :param fine_tuned_answer_retriever: An instance of FineTunedAnswerRetriever for retrieving fine-tuned answers.
        :param hyde_retriever: An instance of HyDEGenerator for generating hypothetical documents.
        :param client: Client for API interactions.
        :param multiquery_generator: An instance of MultiqueryGenerator for generating additional queries.
        :param document_retriever: An instance of DocumentRetriever for retrieving documents.
        :param scoring_method: The method used for scoring ('scoring_agent' or 'reranker').
        :param answer_model: The model used for generating answers (e.g., 'gpt-3.5').
        """
        self.context_scorer = context_scorer
        self.fine_tuned_answer_retriever = fine_tuned_answer_retriever
        self.hyde_retriever = hyde_retriever
        self.client = client
        self.multiquery_generator = multiquery_generator
        self.scoring_method = scoring_method
        self.document_retriever = document_retriever
        self.answer_model = answer_model



    def get_contexts_from_questions(self, question, top_k, num_docs, model, tokenizer):
        """
        Retrieves top contexts from a given question.
        :param question: String, the user's question.
        :param top_k: Integer, the number of top contexts to retrieve.
        :param num_docs: Integer, the number of documents to consider for context retrieval.
        :return: A sorted list of contexts based on relevance scores.
        """
        all_contexts = []
        if self.scoring_method == 'scoring_agent':
            contexts = self.context_scorer.calculate_top_contexts_for_single_question(question, top_k, num_docs)
        elif self.scoring_method == 'reranker':
            contexts = self.context_scorer.calculate_reranked_score_for_single_question(question, model, tokenizer, top_k, num_docs)
        all_contexts = contexts
        sorted_contexts = sorted(all_contexts, key=lambda x: self.extract_relevance_score(x), reverse=True)[:top_k]

        return sorted_contexts


    def get_contexts_from_answers(self, question, num_repeats, top_k_ans, num_docs_ans, model, tokenizer):
        """
        Retrieves relevant contexts based on answers generated for the given question.
        :param question: String, the original question posed by the user.
        :param num_repeats: Integer, the number of times the answer retrieval process is repeated.
        :param top_k_ans: Integer, the number of top contexts to return.
        :param num_docs_ans: Integer, the number of documents to consider when retrieving contexts for answers.
        :return: A list of sorted contexts based on their relevance scores.
        """
        all_contexts = []
        for _ in range(num_repeats):
            selected_answer = self.fine_tuned_answer_retriever.return_finetuned_answer(question)
            if self.scoring_method == 'scoring_agent':
                contexts = self.context_scorer.calculate_top_contexts_for_single_answer(selected_answer, question, num_docs_ans // num_repeats, num_docs_ans // num_repeats)
            elif self.scoring_method == 'reranker':
                contexts = self.context_scorer.calculate_reranked_score_for_single_answer(selected_answer, question, model, tokenizer, num_docs_ans // num_repeats, num_docs_ans // num_repeats)
            all_contexts += contexts
        sorted_contexts = sorted(all_contexts, key=lambda x: self.extract_relevance_score(x), reverse=True)[:top_k_ans]

        return sorted_contexts

    def get_contexts_using_hyde(self, question, num_repeats, top_k_ans, num_docs_ans, model, tokenizer):
        """
        Retrieves contexts based on hypothetical answers generated for the given question using the HyDE method.
        :param question: String, the original question posed by the user.
        :param num_repeats: Integer, the number of times the HyDE generation process is repeated.
        :param top_k_ans: Integer, the number of top contexts to return.
        :param num_docs_ans: Integer, the number of documents to consider when retrieving contexts for HyDE answers.
        :return: A list of sorted contexts based on their relevance scores.
        """
        all_contexts = []
        for _ in range(num_repeats):
            selected_answer = self.hyde_retriever.generate_hyde(question)
            if self.scoring_method == 'scoring_agent':
                contexts = self.context_scorer.calculate_top_contexts_for_single_answer(selected_answer, question, num_docs_ans // num_repeats, num_docs_ans // num_repeats)
            elif self.scoring_method == 'reranker':
                contexts = self.context_scorer.calculate_reranked_score_for_single_answer(selected_answer, question, model, tokenizer, num_docs_ans // num_repeats, num_docs_ans // num_repeats)
            all_contexts += contexts
        sorted_contexts = sorted(all_contexts, key=lambda x: self.extract_relevance_score(x), reverse=True)[:top_k_ans]

        return sorted_contexts

    def get_contexts_using_multiquery(self, questions, top_k, num_docs, model, tokenizer):
        """
        Retrieves contexts based on a set of multiple queries generated from the original question.
        :param questions: List of Strings, original question and its variations.
        :param top_k: Integer, the number of top contexts to return.
        :param num_docs: Integer, the number of documents to consider when retrieving contexts.
        :return: A list of sorted contexts based on their relevance scores.
        """
        all_contexts = []
        if self.scoring_method == 'scoring_agent':
            contexts = self.context_scorer.calculate_scoringagent_score_for_single_question_multiquery(questions, top_k, num_docs)
        elif self.scoring_method == 'reranker':
            contexts = self.context_scorer.calculate_reranked_score_for_single_question_multiquery(questions, model, tokenizer, top_k, num_docs)
        all_contexts = contexts
        sorted_contexts = sorted(all_contexts, key=lambda x: self.extract_relevance_score(x), reverse=True)[:top_k]
        return sorted_contexts

    def extract_relevance_score(self, text):
        """
        Extracts the relevance score from the given text.
        :param text: String, text containing the relevance score.
        :return: Float, the extracted relevance score.
        """
        match = re.search(r"Relevance with the question: (\d+\.\d+)/10", text)
        return float(match.group(1)) if match else 0

    def extract_context(self, text):
        """
        Extracts the context portion from the given text.
        :param text: String, text containing the context and the relevance score.
        :return: String, the extracted context.
        """
        extracted = text.split('Relevance with the question: ')[0]
        return extracted

    def get_combined_top_contexts_with_ft(self, question, num_repeats, top_k, top_k_ans, num_docs, num_docs_ans, final_k, model, tokenizer):
        """
        Combines the top contexts from both question-based and answer-based retrieval methods.
        :param question: String, the original question posed by the user.
        :param num_repeats: Integer, the number of repetitions for answer retrieval.
        :param top_k: Integer, the number of top contexts to return from question-based retrieval.
        :param top_k_ans: Integer, the number of top contexts to return from answer-based retrieval.
        :param num_docs: Integer, the number of documents to consider for question-based retrieval.
        :param num_docs_ans: Integer, the number of documents to consider for answer-based retrieval.
        :param final_k: Integer, the final number of top contexts to return after combining.
        :return: A tuple containing two lists - the final sorted contexts and the removed duplicate contexts.
        """
        unique_top_contexts = []
        removed_contexts = []

        if top_k_ans > 0:
            all_contexts_from_answers = self.get_contexts_using_only_answers(question, num_repeats, top_k_ans, num_docs_ans, model, tokenizer)
            context_generator = UniqueContextGenerator(all_contexts_from_answers)
            unique_top_contexts_from_answers, removed_contexts_from_answers = context_generator.get_unique_contexts()
            unique_top_contexts.extend(unique_top_contexts_from_answers)
            removed_contexts.extend(removed_contexts_from_answers)

        if top_k > 0:  
            top_contexts_from_question = self.get_contexts_using_only_questions(question, top_k, num_docs, model, tokenizer)
            context_generator = UniqueContextGenerator(top_contexts_from_question)
            unique_top_contexts_from_questions, removed_contexts_from_questions = context_generator.get_unique_contexts()
            unique_top_contexts.extend(unique_top_contexts_from_questions)
            removed_contexts.extend(removed_contexts_from_questions)

        #Context extraction
        extracted_contexts = [self.extract_context(context) for context in unique_top_contexts]

        #Remove duplicate
        non_duplicate_contexts = list(set(extracted_contexts))

        final_contexts = []
        used_contexts = set()
        for context in unique_top_contexts:
            extracted = self.extract_context(context)
            if extracted in non_duplicate_contexts and extracted not in used_contexts:
                final_contexts.append(context)
                used_contexts.add(extracted)
        sorted_contexts = sorted(final_contexts, key=lambda x: self.extract_relevance_score(x), reverse=True)[:final_k]
        return sorted_contexts, removed_contexts

    def get_combined_top_contexts_with_ft_multiquery(self, original_question, num_repeats, top_k, top_k_ans, num_docs, num_docs_ans, unique_num, final_k, model, tokenizer):
        """
        Combines the top contexts from both fine-tuned answers and multiquery-generated questions.
        :param original_question: String, the original question posed by the user.
        :param num_repeats: Integer, the number of repetitions for answer retrieval.
        :param top_k: Integer, the number of top contexts to return from question-based retrieval.
        :param top_k_ans: Integer, the number of top contexts to return from answer-based retrieval.
        :param num_docs: Integer, the number of documents to consider for question-based retrieval.
        :param num_docs_ans: Integer, the number of documents to consider for answer-based retrieval.
        :param unique_num: Integer, the number of additional queries to generate for multiquery retrieval.
        :param final_k: Integer, the final number of top contexts to return after combining.
        :return: A tuple containing two lists - the final sorted contexts and the removed duplicate contexts.
        """
        questions = self.multiquery_generator.generate_additional_questions(original_question, unique_num)
        unique_top_contexts = []
        removed_contexts = []

        if top_k_ans > 0:
            all_contexts_from_answers = self.get_contexts_using_only_answers(original_question, num_repeats, top_k_ans, num_docs_ans, model, tokenizer)
            context_generator = UniqueContextGenerator(all_contexts_from_answers)
            unique_top_contexts_from_answers, removed_contexts_from_answers = context_generator.get_unique_contexts()
            unique_top_contexts.extend(unique_top_contexts_from_answers)
            removed_contexts.extend(removed_contexts_from_answers)

        if top_k > 0:  
            contexts_q = self.get_contexts_using_multiquery(questions, top_k, num_docs, unique_num, model, tokenizer)
            context_generator = UniqueContextGenerator(contexts_q)
            unique_top_contexts_from_questions, removed_contexts_from_questions = context_generator.get_unique_contexts()
            unique_top_contexts.extend(unique_top_contexts_from_questions)
            removed_contexts.extend(removed_contexts_from_questions)

        extracted_contexts = [self.extract_context(context) for context in unique_top_contexts]
        non_duplicate_contexts = list(set(extracted_contexts))

        final_contexts = []
        used_contexts = set()
        for context in unique_top_contexts:
            extracted = self.extract_context(context)
            if extracted in non_duplicate_contexts and extracted not in used_contexts:
                final_contexts.append(context)
                used_contexts.add(extracted)

        sorted_contexts = sorted(final_contexts, key=lambda x: self.extract_relevance_score(x), reverse=True)[:final_k]
        return sorted_contexts, removed_contexts

    def get_combined_top_contexts_with_HyDE(self, question, num_repeats, top_k, top_k_ans, num_docs, num_docs_ans, final_k, model, tokenizer):
        """
        Combines the top contexts from the HyDE-generated answers.
        :param question: String, the original question posed by the user.
        :param num_repeats: Integer, the number of repetitions for HyDE answer generation.
        :param top_k: Integer, the number of top contexts to return from question-based retrieval.
        :param top_k_ans: Integer, the number of top contexts to return from HyDE answer-based retrieval.
        :param num_docs: Integer, the number of documents to consider for question-based retrieval.
        :param num_docs_ans: Integer, the number of documents to consider for HyDE answer-based retrieval.
        :param final_k: Integer, the final number of top contexts to return after combining.
        :return: A tuple containing two lists - the final sorted contexts and the removed duplicate contexts.
        """
        unique_top_contexts = []
        removed_contexts = []

        if top_k_ans > 0:
            all_contexts_from_answers = self.get_contexts_using_hyde(question, num_repeats, top_k_ans, num_docs_ans, model, tokenizer)
            context_generator = UniqueContextGenerator(all_contexts_from_answers)
            unique_top_contexts_from_answers, removed_contexts_from_answers = context_generator.get_unique_contexts()
            unique_top_contexts.extend(unique_top_contexts_from_answers)
            removed_contexts.extend(removed_contexts_from_answers)

        if top_k > 0: 
            top_contexts_from_question = self.get_contexts_using_only_questions(question, top_k, num_docs, model, tokenizer)
            context_generator = UniqueContextGenerator(top_contexts_from_question)
            unique_top_contexts_from_questions, removed_contexts_from_questions = context_generator.get_unique_contexts()
            unique_top_contexts.extend(unique_top_contexts_from_questions)
            removed_contexts.extend(removed_contexts_from_questions)

        extracted_contexts = [self.extract_context(context) for context in unique_top_contexts]
        non_duplicate_contexts = list(set(extracted_contexts))

        final_contexts = []
        used_contexts = set()
        for context in unique_top_contexts:
            extracted = self.extract_context(context)
            if extracted in non_duplicate_contexts and extracted not in used_contexts:
                final_contexts.append(context)
                used_contexts.add(extracted)
        sorted_contexts = sorted(final_contexts, key=lambda x: self.extract_relevance_score(x), reverse=True)[:final_k]
        return sorted_contexts, removed_contexts

    def get_contexts_using_only_questions(self, question, top_k, num_docs, model, tokenizer):
        """
        Retrieves contexts based only on the original question, ignoring answers.
        :param question: String, the user's original question.
        :param top_k: Integer, the number of top contexts to retrieve.
        :param num_docs: Integer, the number of documents to consider for context retrieval.
        :return: A sorted list of contexts based on their relevance scores.
        """
        return self.get_contexts_from_questions(question, top_k, num_docs, model, tokenizer)

    def get_contexts_using_only_answers(self, question, num_repeats, top_k_ans, num_docs_ans, model, tokenizer):
        """
        Retrieves contexts based only on the answers generated for the original question.
        :param question: String, the user's original question.
        :param num_repeats: Integer, the number of repetitions for answer generation.
        :param top_k_ans: Integer, the number of top contexts to retrieve for answers.
        :param num_docs_ans: Integer, the number of documents to consider for answer-based context retrieval.
        :return: A sorted list of contexts based on their relevance scores.
        """
        return self.get_contexts_from_answers(question, num_repeats, top_k_ans, num_docs_ans, model, tokenizer)

    def generate_final_template(self, question, combined_top_contexts, default_prompt):
        """
        Generates a final template that combines the question with its top contexts.
        :param question: String, the user's question.
        :param combined_top_contexts: List of contexts combined with their relevance scores.
        :param default_prompt: String, the default prompt to be used in the template.
        :return: A formatted string representing the final template.
        """
        final_template = default_prompt + "\n\n" + "Question: " + question + "\n" + ("-" * 50 + "\n")
        for order, detail in enumerate(combined_top_contexts):
            final_template += "-" * 50 + "\n" + f"Num.{order+1} " + detail + "\n"
        return final_template

    def generate_final_template_without_score(self, question, combined_top_contexts, default_prompt):
        """
        Generates a final template without showing the relevance scores.
        :param question: String, the user's question.
        :param combined_top_contexts: List of contexts.
        :param default_prompt: String, the default prompt to be used in the template.
        :return: A formatted string representing the final template without relevance scores.
        """
        final_template = default_prompt + "\n\n" + "Question: " + question + "\n" + ("-" * 50 + "\n")
        updated_contexts = []
        for order, context in enumerate(combined_top_contexts):
            updated_context = context.split('\n--------------------\nRelevance with the question:')[0]
            updated_contexts.append(updated_context.strip())
        combined_top_contexts = updated_contexts
        for order, detail in enumerate(combined_top_contexts):
            final_template += "-" * 50 + "\n" + f"Num.{order+1} " + detail + "\n"
        return final_template

    def get_answer(self, client, question_example, answer_example, final_template):
        """
        Generates an answer using a specified AI model.
        :param question_example: String, example question to guide the response.
        :param answer_example: String, example answer to guide the response.
        :param final_template: String, the final template to be used for generating the answer
        :return: The generated answer based on the input template and examples.
        """
        if self.answer_model == "gpt-4":
          response = client.chat.completions.create(
              model="gpt-4-1106-preview",
              messages=[
                  {"role": "system", "content": "You are an expert on regulatory guidelines, providing answers and clarity on various topics related to the agency's regulations and recommendations."},
                  {"role": "user", "content": f"{question_example}"},
                  {"role": "assistant", "content": f"{answer_example}"},
                  {"role": "user", "content": f"{final_template}"}
              ]
          )
          answer = response.choices[0].message.content
        elif self.answer_model == "gpt-3.5":
          response = client.chat.completions.create(
              model="gpt-3.5-turbo-1106",
              messages=[
                  {"role": "system", "content": "You are an expert on regulatory guidelines, providing answers and clarity on various topics related to the agency's regulations and recommendations."},
                  {"role": "user", "content": f"{question_example}"},
                  {"role": "assistant", "content": f"{answer_example}"},
                  {"role": "user", "content": f"{final_template}"}
              ]
          )
          answer = response.choices[0].message.content

        return answer

    def get_answer_without_score(self, client, question_example, answer_example, final_template):
        """
        Generates an answer without including relevance scores in the context, using a specified AI model.
        :param question_example: String, example question to guide the response.
        :param answer_example: String, example answer to guide the response.
        :param final_template: String, the final template without relevance scores.
        :return: The generated answer based on the input template and examples.
        """
        if self.answer_model == "gpt-4":
          response = client.chat.completions.create(
              model="gpt-4-1106-preview",
              messages=[
                  {"role": "system", "content": "You are an expert on regulatory guidelines, providing answers and clarity on various topics related to the agency's regulations and recommendations."},
                  {"role": "user", "content": f"{question_example}"},
                  {"role": "assistant", "content": f"{answer_example}"},
                  {"role": "user", "content": f"{final_template}"}
              ]
          )
          answer = response.choices[0].message.content
        elif self.answer_model == "gpt-3.5":
          response = client.chat.completions.create(
              model="gpt-3.5-turbo-1106",
              messages=[
                  {"role": "system", "content": "You are an expert on regulatory guidelines, providing answers and clarity on various topics related to the agency's regulations and recommendations."},
                  {"role": "user", "content": f"{question_example}"},
                  {"role": "assistant", "content": f"{answer_example}"},
                  {"role": "user", "content": f"{final_template}"}
              ]
          )
          answer = response.choices[0].message.content

        return answer


