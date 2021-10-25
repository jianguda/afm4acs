import logging
from datasets import load_metric
from nlgeval import NLGEval

# logger
# Gets or creates a logger
logger = logging.getLogger(__name__)
# set log level
logger.setLevel(logging.INFO)


# define file handler and set formatter
# file_handler = logging.FileHandler('logfile.log')
# formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
# file_handler.setFormatter(formatter)
# add file handler to logger
# logger.addHandler(file_handler)


class Metric:
    @staticmethod
    def report(predictions, references):
        corpus_bleu_score, sentence_bleu_score = Metric.report_bleu(predictions, references)
        meteor_score = Metric.report_meteor(predictions, references)
        rouge_score = Metric.report_rouge(predictions, references)
        # print(f'corpus bleu score: {corpus_bleu_score}')
        # print(f'sentence bleu score: {sentence_bleu_score}')
        # print(f'meteor score: {meteor_score}')
        # print(f'rouge score: {rouge_score}')
        x = f'c_bleu = {corpus_bleu_score} | s_bleu = {sentence_bleu_score} | meteor = {meteor_score} | rouge = {rouge_score}'
        print(x)
        logger.info(x)
        return corpus_bleu_score, sentence_bleu_score, meteor_score, rouge_score

    @staticmethod
    def report_bleu(predictions, references):
        predictions = [prediction.split() for prediction in predictions]
        references = [[reference.split()] for reference in references]
        metric = load_metric('bleu')
        # print(metric.inputs_description)
        corpus_score = metric.compute(predictions=predictions, references=references)['bleu']
        sum_sentence_bleu = 0.0
        for reference, prediction in zip(references, predictions):
            sum_sentence_bleu += metric.compute(predictions=[prediction], references=[reference], smooth=True)['bleu']
        sentence_score = sum_sentence_bleu / len(predictions)
        return round(corpus_score * 100, 2), round(sentence_score * 100, 2)

    @staticmethod
    def report_meteor(predictions, references, version=1.5):
        if version == 1.5:
            # version 1.5 https://aclanthology.org/W14-3348.pdf
            metrics_dict = NLGEval().compute_metrics([references], predictions)
            return round(metrics_dict['METEOR'] * 100, 2)
        else:
            assert version == 1.0
            # version 1.0 https://aclanthology.org/W07-0734.pdf
            predictions = [prediction for prediction in predictions]
            references = [reference for reference in references]
            metric = load_metric('meteor')
            # print(metric.inputs_description)
            scores = metric.compute(predictions=predictions, references=references)
            return round(scores['meteor'] * 100, 2)

    @staticmethod
    def report_rouge(predictions, references):
        predictions = [prediction for prediction in predictions]
        references = [reference for reference in references]
        metric = load_metric('rouge')
        # print(metric.inputs_description)
        scores = metric.compute(predictions=predictions, references=references)
        return round(scores['rougeL'].mid.fmeasure * 100, 2)


if __name__ == '__main__':
    predictions = ["It is a guide to action which ensures that the military always obeys the commands of the party"]
    references = ["It is a guide to action that ensures that the military will forever heed Party commands"]
    Metric.report(predictions, references)
