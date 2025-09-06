import numpy
import torch
import torch.nn as nn

import numpy as np
import scipy

# import spacy

import whisper
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, WhisperForConditionalGeneration, WhisperProcessor, GPT2Tokenizer, GPT2LMHeadModel

from nltk import word_tokenize, sent_tokenize
from pymorphy3 import MorphAnalyzer

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from src.utils import get_speech_and_silence_timestamps, AphasiaDatasetWaveform

import re
import os
import gzip
from collections import Counter


from phonemizer import phonemize


class CarefulWhisper(nn.Module):

    def __init__(self, device="cpu"):
        super(CarefulWhisper, self).__init__()

        # self.whisper = whisper.load_model("antony66/whisper-large-v3-russian")
        self.whisp = WhisperForConditionalGeneration.from_pretrained( "antony66/whisper-large-v3-russian", low_cpu_mem_usage=True, use_safetensors=True)
        self.whisp_processor = WhisperProcessor.from_pretrained("antony66/whisper-large-v3-russian")
        self.whisp_pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.whisp,
            tokenizer=self.whisp_processor.tokenizer,
            feature_extractor=self.whisp_processor.feature_extractor,
            # max_new_tokens=256,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            device=device
        )

        self.options = whisper.DecodingOptions()

        self.w2v_processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-xls-r-1b-russian")
        self.w2v = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-xls-r-1b-russian")

        # self.spacy_vectorizer = spacy.load("ru_core_news_lg")
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

        self.bert_tokenizer = AutoTokenizer.from_pretrained("disk0dancer/ruBert-base-finetuned-pos")
        self.bert_model = AutoModelForTokenClassification.from_pretrained("disk0dancer/ruBert-base-finetuned-pos")

        self.tagger = pipeline("token-classification", model=self.bert_model, tokenizer=self.bert_tokenizer, aggregation_strategy="simple")

        self.device = device

    # Word per seconds. clean_transcription / audio duration
    @staticmethod
    def compute_wps(transcription, audio):
        return len(transcription) / (audio.shape[-1] / 16_000)

    # Percentage time spoken
    @staticmethod
    def compute_ptc(speech_dur, chunk_dur):
        # print(speech_dur, chunk_dur.shape)
        return speech_dur / chunk_dur.shape[-1]

    # Type token ratio
    @staticmethod
    def ttr(text):
        if len(text) == 0:
            return 0
        return len(set(text)) / len(text)

    # Moving average ttr
    @staticmethod
    def mattr(tokens, window_size):
        if len(tokens) < window_size:
            return CarefulWhisper.ttr(tokens)
        return np.mean([
            len(set(tokens[i:i + window_size])) / window_size
            for i in range(len(tokens) - window_size + 1)
        ])

    @staticmethod
    def gzip_ratio(text):
        if len(text) == 0:
            return 0
        compressed = gzip.compress(text.encode('utf-8'))
        return len(compressed) / len(text.encode('utf-8')) if text else 0

    # Hypergeometric Distribution Diversity
    @staticmethod
    def hdd(tokens, sample_size=10):
        if not tokens:
            return 0.0
        freqs = Counter(tokens)
        N = min(len(tokens), sample_size)
        hdd_value = 0.0
        for type_, freq in freqs.items():
            p = 1 - scipy.stats.hypergeom.pmf(0, N, freq, sample_size)
            # print(scipy.stats.hypergeom.pmf(0, N, freq, sample_size))
            hdd_value += p
        # print(hdd_value, sample_size, hdd_value / sample_size)
        return hdd_value / sample_size

    # Measure of Textual Lexical Diversity
    @staticmethod
    def mtld(tokens, ttr_threshold=0.6):
        def _mtld_calc(seq):
            factors = 0
            token_count = 0
            types = set()
            ttr = 1.0
            for i, token in enumerate(seq):
                token_count += 1
                types.add(token)
                ttr = len(types) / token_count
                if ttr <= ttr_threshold:
                    factors += 1
                    token_count = 0
                    types = set()
            if token_count > 0 and ttr < 1.0:
                factors += (1 - ttr_threshold) / (1 - ttr)
            return len(seq) / factors if factors != 0 else 0

        return (_mtld_calc(tokens) + _mtld_calc(tokens[::-1])) / 2

    @staticmethod
    def word_information(lemmatized_tokens):
        if len(lemmatized_tokens) == 0:
            return 0
        freqs = np.array(list(Counter(lemmatized_tokens).values()))
        probs = freqs / freqs.sum()
        return -np.sum(probs * np.log2(probs + 1e-6))

    @staticmethod
    def pos_ratio(tags):
        if len(tags) == 0:
            return np.zeros(3)
        nouns = 0
        verbs = 0
        adjectives = 0
        for token in tags:
            if token['entity_group'] == "N":
                nouns += 1
            elif token['entity_group'][:3] == "Adv":
                adjectives += 1
            elif token['entity_group'][0] == "V" or token['entity_group'] == "Aux":
                verbs += 1
        return np.array([nouns / len(tags), verbs / len(tags), adjectives / len(tags)])

    @staticmethod
    def mean_sent_length(text):
        sentences = sent_tokenize(text)
        if len(sentences) == 0:
            return 0
        return sum([len(x) for x in sentences]) / len(sentences)

    @staticmethod
    def prep_text(text):
        process_text = text.lower()
        # print(process_text)
        process_text = re.sub("[^\w\s]+", " ", process_text)
        process_text = process_text.strip()
        # print(process_text)
        # process_text = re.split(r'\s+', process_text)
        return process_text

    @staticmethod
    def calc_sent_dist(ref, hyp):
        d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
        for i in range(len(ref) + 1):
            d[i][0] = i
        for j in range(len(hyp) + 1):
            d[0][j] = j
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                d[i][j] = min(
                    d[i - 1][j] + 1,  # Deletion
                    d[i][j - 1] + 1,  # Insertion
                    d[i - 1][j - 1] + cost  # Substitution
                )
        return d[len(ref)][len(hyp)]

    @staticmethod
    def wer(ref_text, hyp_text):
        if len(ref_text) == 0 or len(hyp_text) == 0:
            return 1
        ref_text = CarefulWhisper.prep_text(ref_text).split()
        hyp_text = CarefulWhisper.prep_text(hyp_text).split()
        return CarefulWhisper.calc_sent_dist(ref_text, hyp_text) / len(ref_text)

    @staticmethod
    def cer(ref_text, hyp_text):
        if len(ref_text) == 0 or len(hyp_text) == 0:
            return 1
        ref_text = CarefulWhisper.prep_text(ref_text)
        hyp_text = CarefulWhisper.prep_text(hyp_text)
        return CarefulWhisper.calc_sent_dist(ref_text, hyp_text) / len(ref_text)

    # def word_vector_coherence(self, text: str) -> float:
    #     doc = list(self.spacy_vectorizer(text).sents)
    #     sentence_vecs = [sent.vector for sent in doc if sent.vector_norm > 0]
    #
    #     if len(sentence_vecs) < 2:
    #         return 0.0  # not enough for coherence
    #
    #     sims = []
    #     for i in range(len(sentence_vecs) - 1):
    #         a, b = sentence_vecs[i], sentence_vecs[i + 1]
    #         cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    #         sims.append(cosine)
    #
    #     return float(np.mean(sims)) if sims else 0.0

    def gpt2_perplexity(self, text, window_size=512):
        tokens = self.gpt2_tokenizer.encode(text)
        if len(tokens) == 0:
            return 1
        nlls = []
        stride = window_size // 2

        for i in range(0, len(tokens), stride):
            begin = max(i - stride, 0)
            end = min(i + window_size, len(tokens))
            input_ids = torch.tensor([tokens[begin:end]])
            target_ids = input_ids.clone()
            target_ids[:, :-stride] = -100  # mask left context

            with torch.no_grad():
                outputs = self.gpt2_model(input_ids, labels=target_ids)
                log_likelihood = outputs.loss * (end - begin - stride)
                nlls.append(log_likelihood.item())

            if end == len(tokens):
                break

        return float(np.exp(np.sum(nlls) / len(tokens)))

    @staticmethod
    def get_fluency(clean_transcription, acoustic_transcription, tags, audio):
        (speech_dur, speech_count, speech_timestamps, mean_speech_dur,
         silence_dur, silence_count, silence_timestamps, mean_silence_dur) = get_speech_and_silence_timestamps(audio, 16_000, min_silence_duration_ms=300)

        wps = CarefulWhisper.compute_wps(list(clean_transcription.split()), audio)
        ptc = CarefulWhisper.compute_ptc(speech_dur, audio)
        pause_len = silence_dur

        end = 0
        pauses_dists = []
        for ts in silence_timestamps:
            if ts["start"] > 0:
                pauses_dists.append(ts["start"] - end)
            end = ts["end"]
        # print(pauses_dists)
        pauses_dists = np.array(pauses_dists) / 16_000
        if len(pauses_dists) == 0:
            pause_dist_feat = np.zeros(5)
        else:
            pause_dist_feat = numpy.quantile(pauses_dists, [0.1, 0.25, 0.5, 0.75, 0.95])

        if len(clean_transcription) == 0:
            pause_per_word = 0
        else:
            pause_per_word = silence_count / len(clean_transcription)
        #
        # phonemes = phonemize(
        #     clean_transcription,
        #     language='ru',
        #     backend='espeak',
        #     strip=True,
        #     preserve_punctuation=True,
        #     njobs=1
        # ).split()
        # print(tags)
        # phoneme_lengths = np.array([len(x) for i, x in enumerate(phonemes) if tags[i]['entity_group'] == "N"])
        # phoneme_lengths = np.array([len(phon) for x in tags if x["entity_group"] == "N" for phon in phonemize(
        #     x["word"],
        #     language='ru',
        #     backend='espeak',
        #     strip=True,
        #     preserve_punctuation=True,
        #     njobs=1)])
        # phoneme_lengths = []
        # for tag in tags:
        #     if tag["entity_group"] == "N":
        #         phoneme = phonemize(tag["word"], language='ru', backend='espeak',
        #                              strip=True, preserve_punctuation=True, njobs=1).split(" ")
        #         # print(phoneme)
        #         for phoneme_part in phonemes:
        #             phoneme_lengths.append(len(phoneme_part))
        # phoneme_lengths = np.array(phoneme_lengths)
        # if len(phoneme_lengths) == 0:
        #     mean_phoneme_length = 0
        # else:
        #     mean_phoneme_length = np.mean(phoneme_lengths)
        # phonemes_per_second = CarefulWhisper.compute_wps(phonemes, audio)
        # print(list(clean_transcription.split()), phonemes)
        # print([wps, ptc, pause_len, pause_per_word, mean_phoneme_length, phonemes_per_second], pause_dist_feat)
        return np.concatenate([np.array([wps, ptc, pause_len // 16_000, pause_per_word]), pause_dist_feat])

    @staticmethod
    def get_lexical_richness(clean_transcription):
        ttr = CarefulWhisper.ttr(clean_transcription.split())
        mattr = np.array([CarefulWhisper.mattr(clean_transcription.split(), x) for x in [10, 20, 50]])
        gzip_ratio = CarefulWhisper.gzip_ratio(clean_transcription)
        # hdd = CarefulWhisper.hdd(clean_transcription.split())
        mtld = CarefulWhisper.mtld(clean_transcription.split())

        tokens = word_tokenize(clean_transcription)
        morph = MorphAnalyzer(lang='ru')
        lemmas = [morph.parse(x)[0].normal_form for x in tokens]
        word_inf = CarefulWhisper.word_information(lemmas)
        return np.concatenate([[ttr, gzip_ratio, mtld, word_inf], mattr])

    @staticmethod
    def get_syntax_score(clean_transcription, tags):
        pos_ratio = CarefulWhisper.pos_ratio(tags)
        mean_sent_length = CarefulWhisper.mean_sent_length(clean_transcription)
        return np.concatenate([np.array([mean_sent_length, ]), pos_ratio])

    @staticmethod
    def get_pronunciation_score(clean_transcription, acoustic_transcription):
        wer = CarefulWhisper.wer(clean_transcription, acoustic_transcription)
        cer = CarefulWhisper.cer(clean_transcription, acoustic_transcription)
        return np.array([wer, cer])

    def get_coherence_score(self, clean_transcription: str):
        # word_vector_coherence = self.word_vector_coherence(clean_transcription)
        gpt2_perplexity = self.gpt2_perplexity(clean_transcription)

        return np.array([gpt2_perplexity, ])

    def forward(self, audio):

        # mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), n_mels=self.whisper.dims.n_mels).to(self.whisper.device)

        clean_transcription = self.whisp_pipeline(audio.numpy().squeeze(), generate_kwargs={"language": "russian", "max_new_tokens": 256}, return_timestamps=False)["text"] #  whisper.decode(self.whisper, mel, self.options)

        input_values = self.w2v_processor(audio.squeeze(), return_tensors="pt", sampling_rate=16_000).input_values
        logits = self.w2v(input_values).logits.cpu()

        predicted_ids = torch.argmax(logits, dim=-1)
        acoustic_transcription = self.w2v_processor.decode(predicted_ids[0])

        # tokenizer = AutoTokenizer.from_pretrained("disk0dancer/ruBert-base-finetuned-pos")
        # model = AutoModelForTokenClassification.from_pretrained("disk0dancer/ruBert-base-finetuned-pos")

        # print(phonemes)
        # print(clean_transcription)
        # print(acoustic_transcription)
        # pos_tagger = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        # acoustic_tags = pos_tagger(acoustic_transcription)
        clean_tags = self.tagger(clean_transcription)
        # print(tags)

        fluency_features = CarefulWhisper.get_fluency(clean_transcription, acoustic_transcription, clean_tags, audio)
        lexical_features = CarefulWhisper.get_lexical_richness(clean_transcription)
        syntax_features = CarefulWhisper.get_syntax_score(clean_transcription, clean_tags)
        pronunciation_features = CarefulWhisper.get_pronunciation_score(clean_transcription, acoustic_transcription)
        coherence_features = self.get_coherence_score(clean_transcription)

        return np.concatenate([fluency_features, lexical_features, syntax_features, pronunciation_features, coherence_features])

class WhisperW2V(nn.Module):

    def __init__(self, device="cpu"):
        super(WhisperW2V, self).__init__()
        self.whisp = WhisperForConditionalGeneration.from_pretrained( "antony66/whisper-large-v3-russian", low_cpu_mem_usage=True, use_safetensors=True)
        self.whisp_processor = WhisperProcessor.from_pretrained("antony66/whisper-large-v3-russian")
        self.whisp_pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.whisp,
            tokenizer=self.whisp_processor.tokenizer,
            feature_extractor=self.whisp_processor.feature_extractor,
            # max_new_tokens=256,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            device=device
        )

        self.w2v_processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-xls-r-1b-russian")
        self.w2v = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-xls-r-1b-russian")


    def forward(self, audio):
        clean_transcription = self.whisp_pipeline(audio.numpy().squeeze(),
                            generate_kwargs={"language": "russian", "max_new_tokens": 256},
                            return_timestamps=False)["text"]

        input_values = self.w2v_processor(audio.squeeze(), return_tensors="pt", sampling_rate=16_000).input_values
        logits = self.w2v(input_values).logits.cpu()

        predicted_ids = torch.argmax(logits, dim=-1)
        acoustic_transcription = self.w2v_processor.decode(predicted_ids[0])

        return clean_transcription, acoustic_transcription


if __name__ == "__main__":
    DATA_DIR = os.path.join('..', 'data')
    VOICES_DIR = os.path.join(DATA_DIR, 'Voices_wav')
    APHASIA_DIR = os.path.join(VOICES_DIR, 'Aphasia')
    NORM_DIR = os.path.join(VOICES_DIR, 'Norm')

    dataset = AphasiaDatasetWaveform(os.path.join(DATA_DIR, "train_filenames_mc_6.csv"), VOICES_DIR, target_sample_rate=16_000, file_format="wav")
    model = CarefulWhisper("cuda")

    output = model(dataset[0][0])
    print(output)
