import string
import torch

def tokenize(main_string, delimeters={" "},
                          special_tokens={"\\newline",'\n'},
                          split_digits=False,
                          lowercase=False):
    """
    Returns a list of tokens delimeted by the strings contained in the
    delimeters set. Punctuation and whitespace characters are treated
    as individual tokens. Delimeters are not counted as tokens.

    Multiple delimeters in a row are counted as a single delimeter.

    string: str
        Some string to be tokenized
    delimeters: set of str
        the delimeters to be used for tokenization
    special_tokens: set of str
        strings that should be treated as individual tokens. Cannot
        contain delimeting characters.
    split_digits: bool
        if true, strings of digits will be split into individual 
        digit tokens 0-9
    lowercase: bool
        if true, all tokens are lowercased
    """
    tokens = []
    s = ""
    for i,char in enumerate(main_string):
        if s in special_tokens:
            if split_digits and s.isdigit():
                for c in s:
                    tokens.append(c)
            else:
                tokens.append(s)
            s = ""
        
        if char in delimeters:
            if len(s) > 0:
                if split_digits and s.isdigit():
                    for c in s:
                        tokens.append(c)
                else:
                    tokens.append(s)
                    if lowercase:
                        tokens[-1] = tokens[-1].lower()
            s = ""

        elif char.isalnum() or char=="_" or char=="<" or char==">":
            s += char
        elif char in string.whitespace:
            if len(s) > 0:
                if split_digits and s.isdigit():
                    for c in s:
                        tokens.append(c)
                else:
                    tokens.append(s)
                    if lowercase:
                        tokens[-1] = tokens[-1].lower()
            tokens.append(char)
            s = ""
        else:
            if len(s) > 0 and s[-1] != "\\":
                if split_digits and s.isdigit():
                    for c in s:
                        tokens.append(c)
                else:
                    tokens.append(s)
                    if lowercase:
                        tokens[-1] = tokens[-1].lower()
                s = ""
            if char == "\\":
                s += char
            else:
                tokens.append(char)
    if len(s) > 0:
        tokens.append(s)
    return tokens

def group_sentences(document, delimeters={".","!","?"},
                  titles={"dr","mr","mrs","ms","prof"}):
    """
    Groups a document string into a list of sentence strings.
    Sentences are defined by an argued delimeter followed by a
    whitespace character. Handles abbreviations by assuming all
    abbreviations are single, capital characters.

    Input:
        document: str
        delimeters: set of str
            end of sentence characters.
        titles: set of str
            enumerated titles that should be considered.
    Returns:
        sentences: list of str
            a list of all of the sentences in the document.
    """
    sentences = []
    running_s = document[0]
    document = document[1:]
    for i,c in enumerate(document[:-2]):
        running_s += c
        if c in delimeters:
            prob_dec = not document[i+1].isspace()
            other = (not document[i+2].isupper())
            other = other and not document[i-1].isalnum()
            prob_abbrev = document[i-1].isupper()
            prob_title =check_if_title(running_s[:-1],titles)
            if not(prob_dec or prob_abbrev or prob_title or other):
                running_s = running_s.strip()
                if len(running_s) > 0:
                    sentences.append(running_s)
                running_s = ""
    running_s = (running_s+document[-2:]).strip()
    if len(running_s) > 0:
        sentences.append(running_s)
    return sentences

def check_if_title(s, titles):
    """
    A helper function to check if the last word in the string is
    contained within a set of strings.

    s: str
        the string to determine if the last sequence of characters,
        delimeted by whitespace, is in the set `titles`
    titles: set of str
        the titles that need to be compared against
    """
    prob_title = False
    for ws_char in string.whitespace:
        splt = s.strip().split(ws_char)
        idx = -1 if len(splt) > 1 else 0
        title_str = splt[idx].strip().lower()
        prob_title = prob_title or title_str in titles
    return prob_title

def get_sent_arr(document, start_token="<START>",
                           stop_token="<STOP>",
                           delimeters={".","!","?"},
                           titles={"dr","mr","mrs","ms","prof"}):
    """
    Groups a document string into a list of sentence strings and then
    puts them into a matrix with a set sequence length.
    Sentences are defined by an argued delimeter followed by a
    whitespace character. Handles abbreviations by assuming all
    abbreviations are single, capital characters.

    Input:
        document: str
        seq_len: int or None
            if None is argued, the seq_len takes on the value of the
            longest sentence length in the document. Otherwise sentences
            that exceed the seq_len are broken into multiple segments
            each with a start and stop token at the start and end of
            the segments respectively.
        start_token: str
            the token value that should be placed at the beginning of
            each sentence
        stop_token: str
            the token value that should be placed at the end of
            each sentence
        delimeters: set of str
            end of sentence characters.
        titles: set of str
            enumerated titles that should be considered.
    Returns:
        tok_list: list of lists of str
            the result is a 2 dimensional matrix in which each entry in
            the row dimension is a sequence of tokens making up a
            sentence in that row.
    """
    sent_list = group_sentences(document,delimeters=delimeters,
                                         titles=titles)
    tok_list = []
    for i,sent in enumerate(sent_list):
        sent = start_token + " " + sent + " " + stop_token
        toks = tokenize(sent)
        tok_list.append(toks)
    return tok_list

class Tokenizer():
    """
    This class assists in tokenizing the data and converting between
    indices and tokens.
    """
    def __init__(self, X=None, Y=None, tok_X=[],
                                       tok_Y=[],
                                       word2idx=None,
                                       idx2word=None,
                                       split_digits=False,
                                       MASK="<MASK>",
                                       START="<START>",
                                       STOP="<STOP>",
                                       INIT="<INIT>",
                                       seq_len_x=None,
                                       seq_len_y=None,
                                       prepend=False,
                                       append=False,
                                       index=True,
                                       strings=None,
                                       words=set(),
                                       verbose=True):
        """
        X: list of strings
            these are the strings that will be tokenized and indexed.
            if None is argued, the word2idx and idx2word must not be
            None
        Y: list of strings
            these are another list of strings that will be tokenized
            and indexed. the words in Y will be be included in the 
            vocabulary if word2idx and idx2word are None
        tok_X: list of tokens
            these are the strings that will be tokenized and indexed.
            if None is argued, the word2idx and idx2word must not be
            None
        tok_Y: list of tokens
            these are another list of strings that will be tokenized
            and indexed. the words in Y will be be included in the 
            vocabulary if word2idx and idx2word are None
        word2idx: dict
            keys: str
                the words or tokens
            values: int
                the integer indices corresponding to each token
        idx2word: dict
            keys: int
                the integer indices corresponding to each token
            values: str
                the words or tokens corresponding to each index
        split_digits: bool
            option to split each digit into a sequence of individual
            digits 0-9
        MASK: str
            the token to correspond to the null embedding
        START: str
            the token to correspond to the START embedding
        STOP: str
            the token to correspond to the STOP embedding
        INIT: str
            a token that optionally all decoding inputs begin with
        seq_len_x: int or None
            if None, then the maximum length of the tokenized X
            will be used for the X sequence length
        seq_len_y: int or None
            if None, then the maximum length of the tokenized Y
            will be used for the Y sequence length
        prepend: bool
            if true, self.START is prepended to the start of the tokens
        append: bool
            if true, self.STOP is appended to the end of the tokens
        index: bool
            if true, the tokens are also converted to indices
        strings: list of str
            each string in this argued list is included in the
            conversion dictionaries word2idx and idx2word
        words: set of str
            a set of words that should be included in the tokenization
        """
        self.MASK = MASK
        self.START = START
        self.STOP = STOP
        self.INIT = INIT
        self.string_X = X
        self.string_Y = Y
        self.split_digits = split_digits
        if split_digits:
            words |= set([str(i) for i in range(10)])
        else:
            words |= set([str(i) for i in range(100)])

        x_max_len = 0
        y_max_len = 0
        assert len(tok_X) == 0 or X is None
        assert len(tok_Y) == 0 or Y is None
        if X is not None or Y is not None:
            if verbose:
                print("Tokenizing")
            if X is not None:
                tok_X,x_max_len,words = self.tokenize(X,words=words,
                                                        verbose=verbose)
            if Y is not None:
                tok_Y,y_max_len,words = self.tokenize(Y, words=words,
                                                        verbose=verbose)
        self.token_X = tok_X
        self.tok_X = tok_X
        self.token_Y = tok_Y
        self.tok_Y = tok_Y
        if word2idx is None:
            word2idx = {self.MASK:0}
        if idx2word is None:
            idx2word = {0:self.MASK}
        for word in words:
            if word not in word2idx:
                idx = len(word2idx)
                word2idx[word] = idx 
                idx2word[idx] = word
        if strings is not None:
            for s in strings:
                if s not in word2idx:
                    idx = len(word2idx)
                    word2idx[s] = idx
                    idx2word[idx] = s
        if self.START not in word2idx:
            word2idx[self.START] = len(word2idx)
            idx2word[word2idx[self.START]] = self.START
        if self.STOP not in word2idx:
            word2idx[self.STOP] = len(word2idx)
            idx2word[word2idx[self.STOP]] = self.STOP
        if self.INIT not in word2idx:
            word2idx[self.INIT] = len(word2idx)
            idx2word[word2idx[self.INIT]] = self.INIT
        self.INIT_IDX = word2idx[self.INIT]
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.seq_len_x = x_max_len+prepend+append if seq_len_x is None\
                                                        else seq_len_x
        self.X = torch.LongTensor([])
        if len(tok_X) > 0 and index:
            if verbose:
                print("Converting to integer indexes")
            self.X = self.index_tokens(tok_X, self.seq_len_x,
                                              prepend=prepend,
                                              append=append)
        self.seq_len_y = y_max_len+prepend+append if seq_len_y is None\
                                                        else seq_len_y
        self.Y = torch.LongTensor([])
        if len(tok_Y) > 0 and index:
            if verbose:
                print("Converting to integer indexes")
            self.Y = self.index_tokens(tok_Y, self.seq_len_y,
                                              prepend=prepend,
                                              append=append)
        self.inits = [self.INIT for i in range(self.seq_len_y)]
        self.inits = self.index_tokens([self.inits],
                                            self.seq_len_y,
                                            prepend=prepend,
                                            append=False)

    def tokenize(self, lostr, words=None, split_digits=None,
                                          verbose=False):
        """
        lostr: list of str
            a list of strings to be tokenized
        words: optional set of str
            the current word set. if none, a new set is created
        split_digits: bool or None
            if none, self.split_digits is used
        """
        max_len = 0
        if words is None:
            words = set()
        if split_digits is None:
            split_digits = self.split_digits
        toks = []
        for i in range(len(lostr)):
            try:
                toks.append(tokenize(lostr[i],
                            split_digits=split_digits))
            except:
                print(i)
                print(lostr[i])
                assert False
            words = words | set(toks[i])
            max_len = max(max_len,len(toks[i]))
            if verbose:
                print(round(float(i)/len(lostr)*100),"%", end="    \r")
        return toks,max_len,words

    def index_tokens(self, toks, seq_len, prepend=False,
                                          append=False,
                                          verbose=True):
        """
        Used to convert tokens to indices

        toks: list of lists of str (N, variable)
            the tokens to be indexed
        seq_len: int
            the length of the sequence. if prepend or append is true,
            they will not add to this number. prepend and append will
            simply replace the tokens at the first and last locations
            in the sequence if there is not enough room for the whole
            sequence.
        prepend: bool
            if true, self.START is prepended to the start of the tokens.
            will potentially overwrite last token if seq_len is not
            long enough to contain all tokens
        append: bool
            if true, self.STOP is appended to the end of the tokens
            will potentially overwrite last token if seq_len is not
            long enough to contain all tokens

        Returns:
            X: torch long tensor (N,XLen)
        """
        if seq_len is None:
            seq_len = len(toks[0])+prepend+append
        X = torch.zeros(len(toks),seq_len).long()
        for i in range(len(toks)):
            if prepend: X[i,0] = self.word2idx[self.START]
            for j,x in enumerate(toks[i]):
                if j < seq_len-prepend:
                    try:
                        X[i,j+prepend] = self.word2idx[x]
                    except:
                        s = "Key error using {}, adding {} to dicts"
                        print(s.format(x,x))
                        idx = len(self.word2idx)
                        self.word2idx[x] = idx
                        self.idx2word[idx] = x
                        X[i,j+prepend] = self.word2idx[x]
            if append:
                X[i,min(j+1+prepend,seq_len-1)]=self.word2idx[self.STOP]
            if verbose:
                print(round(float(i)/len(toks)*100),"%", end="    \r")
        return X

