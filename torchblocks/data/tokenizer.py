"""Tokenization classes."""

from __future__ import absolute_import, division, print_function, unicode_literals

import contextlib
import re
import collections
import logging
import unicodedata
from io import open
from ..utils.common import truncate_sequences, is_string, lowercase_and_normalize

logger = logging.getLogger(__name__)


def load_vocab(dict_path, simplified=False, startswith=None):
    """加载词典文件到dict"""
    token_dict = collections.OrderedDict()
    index = 0
    with open(dict_path, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            token_dict[token] = index
            index += 1

    if not simplified:
        return token_dict
    new_token_dict, keep_tokens = {}, []
    startswith = startswith or []
    for t in startswith:
        new_token_dict[t] = len(new_token_dict)
        keep_tokens.append(token_dict[t])

    for t, _ in sorted(token_dict.items(), key=lambda s: s[1]):
        if t not in new_token_dict and not Tokenizer._is_redundant(t):
            new_token_dict[t] = len(new_token_dict)
            keep_tokens.append(token_dict[t])

    return new_token_dict, keep_tokens


def whitespace_tokenize(text):
    """去除文本中的空白符"""
    text = text.strip()
    return [] if not text else text.split()


class TokenizerBase(object):
    """分词器基类
    """

    def __init__(
            self,
            token_start='[CLS]',
            token_end='[SEP]',
            token_unk='[UNK]',
            token_pad='[PAD]',
            token_mask='[MASK]',
            pre_tokenize=None,
            token_translate=None
    ):
        """参数说明：
        token_unk:
            未知词标记
        token_end:
            句子切分标记，当只有一句话作为输入时，此标记知识作为结束符；当有两句话作为输入时，此标记作为分隔符、最后一句话的结束符
        pad_token:
            padding填充标记
        token_start:
            分类标记，位于整个序列的第一个
        mask_token:
            mask标记
        pre_tokenize：外部传入的分词函数，用作对文本进行预分词。如果传入
                      pre_tokenize，则先执行pre_tokenize(text)，然后在它
                      的基础上执行原本的tokenize函数；
        token_translate：映射字典，主要用在tokenize之后，将某些特殊的token
                         替换为对应的token。
        """
        self._token_pad = token_pad
        self._token_unk = token_unk
        self._token_mask = token_mask
        self._token_start = token_start
        self._token_end = token_end

        self._pre_tokenize = pre_tokenize
        self._token_translate = token_translate or {}
        self._token_translate_inv = {
            v: k
            for k, v in self._token_translate.items()
        }

    def tokenize(self, text, maxlen=None):
        """分词函数
        """
        tokens = [
            self._token_translate.get(token) or token
            for token in self._tokenize(text)
        ]
        if self._token_start is not None:
            tokens.insert(0, self._token_start)
        if self._token_end is not None:
            tokens.append(self._token_end)

        if maxlen is not None:
            index = int(self._token_end is not None) + 1
            truncate_sequences(maxlen, -index, tokens)

        return tokens

    def token_to_id(self, token):
        """token转换为对应的id
        """
        raise NotImplementedError

    def tokens_to_ids(self, tokens):
        """token序列转换为对应的id序列
        """
        return [self.token_to_id(token) for token in tokens]

    def encode(
            self,
            first_text,
            second_text=None,
            maxlen=None,
            pattern='S*E*E',
            truncate_from='right'
    ):
        """输出文本对应token id和segment id
        """
        if is_string(first_text):
            first_tokens = self.tokenize(first_text)
        else:
            first_tokens = first_text

        if second_text is None:
            second_tokens = None
        elif is_string(second_text):
            second_tokens = self.tokenize(second_text)
        else:
            second_tokens = second_text

        if maxlen is not None:
            if truncate_from == 'right':
                index = -int(self._token_end is not None) - 1
            elif truncate_from == 'left':
                index = int(self._token_start is not None)
            else:
                index = truncate_from
            if second_text is not None and pattern == 'S*E*E':
                maxlen += 1
            truncate_sequences(maxlen, index, first_tokens, second_tokens)

        first_token_ids = self.tokens_to_ids(first_tokens)
        first_segment_ids = [0] * len(first_token_ids)

        if second_text is not None:
            if pattern == 'S*E*E':
                idx = int(bool(self._token_start))
                second_tokens = second_tokens[idx:]
            second_token_ids = self.tokens_to_ids(second_tokens)
            second_segment_ids = [1] * len(second_token_ids)
            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)

        return first_token_ids, first_segment_ids

    def id_to_token(self, i):
        """id序列为对应的token
        """
        raise NotImplementedError

    def ids_to_tokens(self, ids):
        """id序列转换为对应的token序列
        """
        return [self.id_to_token(i) for i in ids]

    def decode(self, ids):
        """转为可读文本
        """
        raise NotImplementedError

    def _tokenize(self, text):
        """基本分词函数
        """
        raise NotImplementedError


class Tokenizer(TokenizerBase):
    """Bert原生分词器
    """

    def __init__(self, token_dict, do_lower_case=True, do_basic_tokenize=True, do_tokenize_unk=False, **kwargs):
        """
        参数:
            token_dict:
                词典文件
            do_lower_case:
                是否转换成小写
            do_basic_tokenize:
                分词前，是否进行基础的分词
            do_tokenize_unk:
                分词后，是否生成[UNK]标记，还是在encode阶段生成
        """
        super(Tokenizer, self).__init__(**kwargs)
        if is_string(token_dict):
            token_dict = load_vocab(token_dict)
        self._do_lower_case = do_lower_case
        self._vocab_size = len(token_dict)
        self._token_dict = token_dict
        self._token_dict_inv = {v: k for k, v in token_dict.items()}
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case, never_split=(self._token_unk, self._token_end, self._token_pad, self._token_start, self._token_mask))

        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self._token_dict, unk_token=self._token_unk, do_tokenize_unk=do_tokenize_unk)

        for token in ['pad', 'unk', 'mask', 'start', 'end']:
            with contextlib.suppress(Exception):
                _token_id = token_dict[getattr(self, f'_token_{token}')]
                setattr(self, f'_token_{token}_id', _token_id)

    def _tokenize(self, text, pre_tokenize=True):
        """基本分词函数
        """
        # 以下pre_tokenizer逻辑参考bert4keras
        if self._do_lower_case:
            text = lowercase_and_normalize(text)

        if pre_tokenize and self._pre_tokenize is not None:
            tokens = []
            for token in self._pre_tokenize(text):
                if token in self._token_dict:
                    tokens.append(token)
                else:
                    tokens.extend(self._tokenize(token, False))
            return tokens

        # 以下逻辑参考pytorch版本bert分词器自己的
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text):
                split_tokens.extend(iter(self.wordpiece_tokenizer.tokenize(token)))
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def token_to_id(self, token):
        """token转为vocab中的id"""
        return self._token_dict.get(token, self._token_unk_id)

    def id_to_token(self, id):
        """id转为词表中的token"""
        return self._token_dict_inv[id]

    def decode(self, ids, tokens=None):
        """转为可读文本
        """
        tokens = tokens or self.ids_to_tokens(ids)
        tokens = [token for token in tokens if not self._is_special(token)]
        text, flag = '', False
        for i, token in enumerate(tokens):
            if token[:2] == '##':
                text += token[2:]
            elif len(token) == 1 and self._is_cjk_character(token):
                text += token
            elif len(token) == 1 and self._is_punctuation(token):
                text += token
                text += ' '
            elif i > 0 and self._is_cjk_character(text[-1]):
                text += token
            else:
                text += ' '
                text += token
        text = re.sub(' +', ' ', text)
        text = re.sub("\' (re|m|s|t|ve|d|ll) ", "\'\\1 ", text)
        punctuation = self._cjk_punctuation() + '+-/={(<['
        punctuation_regex = '|'.join([re.escape(p) for p in punctuation])
        punctuation_regex = f'({punctuation_regex}) '
        text = re.sub(punctuation_regex, '\\1', text)
        text = re.sub('(\d\.) (\d)', '\\1\\2', text)
        return text.strip()

    @staticmethod
    def stem(token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        return token[2:] if token[:2] == '##' else token

    @staticmethod
    def _is_space(ch):
        """空格类字符判断
        """
        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
               unicodedata.category(ch) == 'Zs'

    @staticmethod
    def _is_punctuation(ch):
        """标点符号类字符判断（全/半角均在此内）
        提醒：unicodedata.category这个函数在py2和py3下的
        表现可能不一样，比如u'§'字符，在py2下的结果为'So'，
        在py3下的结果是'Po'。
        """
        code = ord(ch)
        return 33 <= code <= 47 or \
               58 <= code <= 64 or \
               91 <= code <= 96 or \
               123 <= code <= 126 or \
               unicodedata.category(ch).startswith('P')

    @staticmethod
    def _cjk_punctuation():
        return u'\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\u00b7\uff01\uff1f\uff61\u3002'

    @staticmethod
    def _is_cjk_character(ch):
        """CJK类字符判断（包括中文字符也在此列）
        参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        """
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
               0x3400 <= code <= 0x4DBF or \
               0x20000 <= code <= 0x2A6DF or \
               0x2A700 <= code <= 0x2B73F or \
               0x2B740 <= code <= 0x2B81F or \
               0x2B820 <= code <= 0x2CEAF or \
               0xF900 <= code <= 0xFAFF or \
               0x2F800 <= code <= 0x2FA1F

    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    @staticmethod
    def _is_redundant(token):
        """判断该token是否冗余（默认情况下不可能分出来）
        """
        if len(token) > 1:
            for ch in Tokenizer.stem(token):
                if (
                        Tokenizer._is_cjk_character(ch) or
                        Tokenizer._is_punctuation(ch)
                ):
                    return True

    def rematch(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系
        """
        if self._do_lower_case:
            text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = lowercase_and_normalize(ch)
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            token = token.lower()
            if token == '[unk]':
                token_mapping.append(char_mapping[offset: offset + 1])
                offset = offset + 1
            elif self._is_special(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start: end])
                offset = end

        return token_mapping


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self,
                 do_lower_case=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """Constructs a BasicTokenizer.
        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        """文本切分成token"""
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        return whitespace_tokenize(" ".join(split_tokens))

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.extend((" ", char, " "))
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all the other languages.
        return (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (
                0x20000 <= cp <= 0x2A6DF) or (0x2A700 <= cp <= 0x2B73F) or (
                0x2B740 <= cp <= 0x2B81F) or (0x2B820 <= cp <= 0x2CEAF) or (
                0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F)

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100, do_tokenize_unk=False):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.do_tokenize_unk = do_tokenize_unk

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.
        Returns:
          A list of wordpiece tokens.
        """
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = f"##{substr}"
                    if substr in self.vocab or not self.do_tokenize_unk:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            if self.do_tokenize_unk and is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    if char in [" ", "\t", "\n", "\r"]:
        return True
    cat = unicodedata.category(char)
    return cat == "Zs"


def _is_control(char):
    """Checks whether `chars` is a control character."""
    if char in ["\t", "\n", "\r"]:
        return False
    cat = unicodedata.category(char)
    return bool(cat.startswith("C"))


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    cat = unicodedata.category(char)
    return bool(cat.startswith("P"))


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError(f"Unsupported string type: {type(text)}")


class SpTokenizer(TokenizerBase):
    """基于SentencePiece模型的封装，使用上跟Tokenizer基本一致。
    """

    def __init__(self, sp_model_path, **kwargs):
        super(SpTokenizer, self).__init__(**kwargs)
        import sentencepiece as spm
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(sp_model_path)
        self._token_pad = self.sp_model.id_to_piece(self.sp_model.pad_id())
        self._token_unk = self.sp_model.id_to_piece(self.sp_model.unk_id())
        self._vocab_size = self.sp_model.get_piece_size()
        for token in ['pad', 'unk', 'mask', 'start', 'end']:
            with contextlib.suppress(Exception):
                _token = getattr(self, f'_token_{token}')
                _token_id = self.sp_model.piece_to_id(_token)
                setattr(self, f'_token_{token}_id', _token_id)

    def token_to_id(self, token):
        """token转换为对应的id
        """
        return self.sp_model.piece_to_id(token)

    def id_to_token(self, i):
        """id转换为对应的token
        """
        return self.sp_model.id_to_piece(i) if i < self._vocab_size else ''

    def decode(self, ids):
        """转为可读文本
        """
        tokens = [
            self._token_translate_inv.get(token) or token
            for token in self.ids_to_tokens(ids)
        ]
        text = self.sp_model.decode_pieces(tokens)
        return convert_to_unicode(text)

    def _tokenize(self, text):
        """基本分词函数
        """
        if self._pre_tokenize is not None:
            text = ' '.join(self._pre_tokenize(text))

        return self.sp_model.encode_as_pieces(text)

    def _is_special(self, i):
        """判断是不是有特殊含义的符号
        """
        return self.sp_model.is_control(i) or \
               self.sp_model.is_unknown(i) or \
               self.sp_model.is_unused(i)

    def _is_decodable(self, i):
        """判断是否应该被解码输出
        """
        return (i < self._vocab_size) and not self._is_special(i)
