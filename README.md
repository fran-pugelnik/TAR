Introduction
============

The amount and quality of data greatly affects the performance of any
model in any NLP task, including coreference resolution. Instead of
collecting more data, it might be more convenient to apply data
augmentation to already obtained data to generate additional, synthetic
data and make the model generalize better. In this paper, we describe an
application of simple data augmentation techniques for improving
performance on coreference resolution task. The model chosen for
tackling the coreference resolution task is an end-to-end neural model
by @lee-etal-2017-end. Its key idea is to consider all spans in a
document as potential mentions to later produce the most likely correct
clustering for mentions.

The advantage of using a neural model for coreference resolution is the
usage of word embeddings to capture the similarity between words. This
can lead to the prediction of false-positive links when the model
conflates paraphrasing with kinship or similarity @lee-etal-2017-end. We
dive deeper into this problem in Section [sec:approach] of this paper.

Related Work
============

Manipulations in the input data in NLP tasks often result in system
failure, although they would not affect human performance on the same
task. Our idea is based on many recent papers that expose the benefits
of data augmentation techniques in such cases, and for natural language
processing tasks in general. A lot of existing work focuses on using
data augmentation for mitigating gender bias in NLP tasks
[@zmigrod-2019; @zhao-etal-2018-gender; @lu-2018].

Some similar approaches were explored for improving syntactic parsing
[@elkahky-etal-2018-challenge], as well as natural language inference
(NLI) @min-etal-2020-syntactic and NLI and sentiment analysis
@kaushik2020Learning. Contextual data augmentation is described in
[@wulv-2018; @kobayashi-2018-contextual] on various text classification
tasks, and other data augmentation operations such as synonym
replacement, random insertion, random swap and random deletion are used
in @EDA-2019 for text classification tasks.

Other interesting techniques, such as back-translation and word
replacing with TF-IDF are discussed in @xie-2019 also for text
classification tasks. Augmentation of a word by a contextual mixture of
multiple related words is used in @gao-etal-2019-soft for the task of
machine translation. @wu-2019 explored the use of existing question
answering datasets as data augmentation for coreference resolution. For
the task of named entity recognition, various approaches such as
label-wise token replacement, synonym replacement, mention replacement,
and shuffle within segments are described in @daiadel-2020. The most
natural choice for data augmentation – replacing the words with their
synonyms is also used in @zhang-2015 for controlling generalization
error.

Neural models often perform well by using superficial features, rather
than more general ones that are preferred and more natural to humans.
Data augmentation is used in @jha-2020 by generating training examples
to encourage the model to focus on the strong features.

Considering the previous success of data augmentation techniques on a
broad spectrum of topics we think that it might be useful to apply
synonym-based augmentation on coreference resolution task in
conversational text.

Database {#sec:data}
========

The data collection created by @chen-choi-2016-character consists of
transcripts of the first two seasons from the TV show *Friends*[^1].
Most of the dialogues between the characters are everyday conversations
and introduce over 200 speakers. Figure [fig:figure1] shows an example
of a multiparty dialogue present in the dataset. The mention “mom” is
not one of the speakers; nonetheless, it refers to the specific person,
Judy, that could appear in some other dialogue. Identifying and
clustering such mentions might require cross-document coreference
resolution.

![Dialogue example with labeled mentions assigned to entities. Taken
from
<https://competitions.codalab.org/competitions/17310>](character-identification-diagram.png "fig:")
[fig:figure1]

The dataset follows the CoNLL 2012 Shared Task data format[^2]. The data
we used consists of documents, each document is delimited and each
episode is considered a document. Each word is associated with a word
form, part-of-speech tag, and its lemma. A speaker is annotated for each
sentence in the training set, as well as in the test set. Each mention
is annotated with a belonging entity ID that is consistent across all
documents. For example, Table [tab:table1][^3] shows a specific sentence
from the dataset. The sentence is from the second episode in season 2,
the scene number is 0 and the speaker is Chandler Bing. The entity of
the mention in this example belongs to their neighbour, the character
Ugly Naked Guy.

[tab:wide-table]

<span>lccllllcc</span> Document ID & Scene ID & Token ID & Word form &
POS tag & Lemma & Speaker & Entity ID\
 & $0$ & $0$ & Ugly & JJ & ugly & &\
 & $0$ & $1$ & Naked & JJ & naked & &\
 & $0$ & $2$ & Guy & NNP & guy & &\
 & $0$ & $3$ & got & VBD & get & &\
 & $0$ & $4$ & a & DT & a & &\
 & $0$ & $5$ & Thighmaster & NN & thighmaster & &\
 & $0$ & $6$ & ! & . & ! & &\

[tab:table1]

Our Approach {#sec:approach}
============

Our experiment consists of an end-to-end neural model by
@lee-etal-2017-end which will be trained on a semeval 2018 dataset and
its augmented counterpart.

Model’s Weaknesses {#subsec:model}
------------------

In the model we used, @lee-etal-2017-end The spans that are not likely
to appear in the clusters are discarted. For the spains remaining after
this step the model decides whether the span is coreferent with some of
the antecedent spans. It returns the resulting coreference links which
(after applying transitivity) imply clustering of spans for the given
document. This model uses word embeddings to capture the similarity
between words. This can lead to a prediction of false positives when the
model confuses paraphrasing with relatedness or similarity. The example
given to illustrate this scenario was when model falsely predicted a
link between a pilot and a flight attendant or Prince Charles and his
wife Camilla and Charles and Diana. Another concern is that the model
sometimes confuses in rarely occurring patterns and examples that humans
wouldn’t find challenging.

Idea
----

It is assumed that both of the problems explained in [subsec:model]
could be mitigated with a larger corpus of training data which would
overcome the sparsity of these patterns.

We argue that using data augmentation to extend our corpus would help
the model generalize and improve its results on patterns like these
while it wouldn’t affect the rest of its performance.

For creating a larger dataset we augmented certain words from the
existing data set and appended it to the original, which is described in
Section [subsec:augmentation]. One can say that by augmenting the
training data with synonyms, we would create duplicate data. However,
that is not the case because synonyms from WordNet often don’t match the
exact synonyms of words from the dataset. As an example, the word
’date’, which meant an appointment to meet someone, was in one case
replaced with the word ’engagement’ which is arguably not a synonym in
this context. This is how the noise is added to our corpus and how the
better generalization of the model is accomplished.

Data Augmentation {#subsec:augmentation}
-----------------

Our approach consists of using data augmentation techniques for
improving AllenNLP’s model’s performance on coreference resolution task.
*nlpaug*[^4] is a library for textual augmentation in machine learning
experiments. The experiment consists of using the augmenter from
WordNet[^5] lexical database, which replaces words from the input
dataset with its synonyms. More specifically, lemmas of words were
replaced with their synonyms. An example of different sentences
generated from the same sentence using synonym augmentation is presented
in Table [tab:augmentation].

[tab:augmentation]

<span>ll</span> ORIG.& Sounds like a date to me.\
AUGM.& Sound like a particular date to me.\
& Speech sound like a engagement to me.\
& Sounds comparable a engagement to me.\

Lemmas that were excluded from the synonym replacement task are
stopwords from Nltk[^6] (Natural language toolkit). Additionally, to
assure that words that greatly affect our task of coreference resolution
stay the same, entities (mostly names of the characters) were also
excluded from synonym replacement operation. Due to limitations imposed
by the .conll file format, it is decided that if a particular synonym
does not consist of one word, but is rather a phrase, it will not be a
candidate for replacement. The words whose lemmas were replaced by
synonyms, had their *Word form* also replaced with the same lemma form.
Having that in mind, some information in the created dataset was lost.
However, The POS and constituency tags from .conll file were not changed
so the replacement of the *Word form* shouldn’t cause a lack of too much
information. After creating such augmentation, it was simply appended to
the existing dataset and presented to the model.

Experimental Setup
==================

The pretrained model in AllenNLP library was trained on the English
coreference resolution data from the CoNLL-2012 shared task
[@pradhan-etal-2012-conll]. We tested the model’s performance on data
described in Section [sec:data], as well as the augmented data described
in Section [subsec:augmentation].

Inspired by @daiadel-2020, we simulate a low-resource setting and select
the first 16 and 32 episodes from the training set to create the
corresponding small and medium training sets to perform data
augmentation and observe the results. We expect to get the best results
by training the model on the largest train set.

For each training set (small, medium, complete) we conduct simple
experiments. We split the training set using random seeds on training
and validation sets (87.5% - 12.5%). The reason for choosing the unusual
train-dev split sizes is because, in this way, an integer division of
episodes in the training set is ensured. We then apply data augmentation
on the training set and test the model performance on the test set
(which is always the same). This experiment is repeated 5 times for each
training set size using different random seeds so statistical
evaluations could be performed.

We then augmented each of the training sets by adding synonym
equivalents for 50% of the training set. The final result were 6
training sets - small, medium, complete, and their corresponding
augmented sets.

Results
=======

The results are presented in Table [tab:results]. The table shows
macro-averaged metrics (precision, recall, and F1-score) obtained from
testing the model trained on a specific training set (S - small, M -
medium, F - full/complete).

[tab:table2]

<span>lccccccccc</span> & & <span>Precision</span> & & &
<span>Recall</span> & & & <span>F1-score</span>\
& S & M & F & S & M & F & S & M & F\
TRAIN SET\
original & 0.448 & 0.490 & 0. & 0.155 & 0.302 & & 0.178 & 0.283\
synonym augmented & 0.502 & 0. & 0. & 0.335\
VALIDATION SET\
original & 0.339 & 0.331 & 0. & 0.281 & 0.242 & & 0.261 & 0.226\
synonym augmented & 0. & 0. & 0.\
TEST SET\

original & 0.564 & 0.581 & 0.579 & 0.431 & 0.396 & 0.318 & **0.480** &
**0.453** & 0.371\
synonym augmented & 0.546 & 0.553 & 0.581 & 0.327 & 0.301 & 0.224 &
0.395 & 0.356 & 0.285\

[tab:results]

<span>lccccccccc</span> & & S & & & M & & & F &\
<span>Data</span> & <span>Precision</span> & <span>Recall</span> &
<span>F1-score</span> & <span>Precision</span> & <span>Recall</span> &
<span>F1-score</span> & <span>Precision</span> & <span>Recall</span> &
<span>F1-score</span>\

TRAIN SET\
original & 0.448 & 0.155 & 0.178 & 0.490 & 0.302 & 0.283\
synonym augmented & 0.502 & 0.335 & 0.\
VALIDATION SET\
original & 0.339 & 0.281 & 0.261 & 0.331 & 0.242 & 0.226\
synonym augmented & 0. & 0. & 0.\
TEST SET\

original & 0.564 & 0.431 & **0.480** & 0.581 & 0.396 & **0.453** & 0.579
& 0.318 & 0.371\
synonym augmented & 0.546 & 0.327 & 0.395 & 0.553 & 0.301 & 0.356 &
0.581 & 0.224 & 0.285\

Analysis
--------

As we were unable to perform testing on a large number of training
instances (5 runs with different random seeds) due to the
time-complexity of the training process, we can’t say much about the
distribution of test results. However, the observation distribution
pairs (e.g. small original - small augmented) are similar in shape, so
we ran a non-parametric Mann-Whitney U test in place of an unpaired
t-test. We wanted to test whether the observations (specifically
F1-scores) in one sample tend to be larger than observations in the
other. Considering the fact that the mean results shown in
Table [tab:results] fluctuate in different ways, we state specific
research hypotheses for each training set size and perform a test
calculation at a significance level of 0.05.

For both small and medium data set, the research hypothesis we chose to
state is that a randomly selected F1-score obtained from the population
of tests conducted on non-augmented data is greater than the same score
obtained when testing on augmented data. Informally, after seeing the
results presented in Table [tab:results], we decided to argue that the
results are significantly better when using non-augmented data. The null
hypothesis is rejected in favor of the alternative. Therefore, it
appears that this type of data augmentation impairs this models
performance.

As for the full data set, according to the mean values of F1-scores, we
chose to test the hypothesis that a randomly selected F1-score obtained
from the population of tests conducted on non-augmented data differs in
any way from F1-score obtained when testing on augmented data. The
resulting p-value equals 0.15, therefore the null hypothesis cannot be
rejected. We can conclude that no significant difference can be
confirmed between the distributions of F1-scores from testing the model
on the full data set and the full augmented data set.

Discussion
----------

The results show that data augmentation with synonyms doesn’t improve
the performance of the coreference resolution model. Moreover, all
subsets of the augmented datasets show lower performance on the test set
than the corresponding subsets of non-augmented data. We argue that this
is the result of the generalization problem presented in
Section [subsec:model]. Obviously, the polysemy of words has created
more problems than benefits for the model. A possible improvement is to
use contextual word embeddings in combination with WordNet synonyms for
augmenting data so the error due to the lack of context lapses. Another
idea would be to include word or span representations that can
distinguish between equivalence and paraphrasing. Moreover, some
generalization techniques should be applied. In terms of other
augmentation methods, antonym replacement or random word generation
might help with generalization.

Secondly, we can see that the obtained results have consistently shown
that the model achieves a lot lower recall than precision. Knowing that
recall denotes how often our model has classified data correctly in
regards to the full set of relevant results, we can notice that the
model has more problems with recognizing phrases that refer to some
entity than it does with correctly deciding on which entity that phrase
is referring to. Therefore, it might be interesting to look into the
reasons for such phenomena in future work.

Conclusion
==========

We presented a simple data augmentation technique for tackling the task
of coreference resolution in multiparty dialogues. We showed that
synonym replacement for this specific data and model showed no
statistically significant improvement in regards to non-augmented data.

Coreference resolution is a challenging NLP problem. The model had
problems detecting mentions which can be seen from the low recall
scores. While precision scores are better, this model wouldn’t be useful
in real-world application and human performance on this task is still
significantly better.

Citations
---------

See @EDA-2019.

@daiadel-2020

@clark-manning-2016-deep

@jha-2020

@lee-etal-2017-end

@zhang-2015

@xie-2019

@wu-2019

@zmigrod-2019

@zhao-etal-2018-gender

@lu-2018

@nie-etal-2020-adversarial

@kaushik2020Learning

@wulv-2018

@gao-etal-2019-soft

Extent of the paper
===================

The paper should have a minimum of 3 and a maximum of 4 pages, plus an
additional page for references.

First subsection {#sec:first}
----------------

This is a subsection of the second section.

Second subsection
-----------------

This is the second subsection of the second section. Referencing the
(sub)sections in text is performed as follows: “in Section [sec:first]
we have shown …”. ...

Figures and tables
==================

Figures
-------

Here is an example on how to include figures in the paper. Figures are
included in LaTeX code immediately *after* the text in which these
figures are referenced. Allow LaTeX to place the figure where it
believes is best (usually on top of the page of at the position where
you would not place the figure). Figures are referenced as follows:
“Figure [fig:figure1] shows …”. Use tilde (`~`) to prevent separation
between the word “Figure” and its enumeration.

![This is the figure caption. Full sentences should be followed with a
dot. The caption should be placed *below* the figure. Caption should be
short; details should be explained in the text.](drawing.pdf "fig:")
[fig:figure1]

Tables
------

There are two types of tables: narrow tables that fit into one column
and a wide table that spreads over both columns.

### Narrow tables

Table [tab:narrow-table] is an example of a narrow table. Do not use
vertical lines in tables – vertical tables have no effect and they make
tables visually less attractive. We recommend using *booktabs* package
for nicer tables.

[tab:narrow-table]

<span>ll</span> Heading1 & Heading2\
One & First row text\
Two & Second row text\
Three & Third row text\
& Fourth row text\

Wide tables
-----------

Table [tab:wide-table] is an example of a wide table that spreads across
both columns. The same can be done for wide figures that should spread
across the whole width of the page.

[tab:wide-table]

<span>llr</span> Heading1 & Heading2 & Heading3\
A & A very long text, longer that the width of a single column & $128$\
B & A very long text, longer that the width of a single column & $3123$\
C & A very long text, longer that the width of a single column & $-32$\

Math expressions and formulas
=============================

Math expressions and formulas that appear within the sentence should be
written inside the so-called *inline* math environment: $2+3$,
$\sqrt{16}$, $h(x)=\mathbf{1}(\theta_1 x_1 + \theta_0>0)$. Larger
expressions and formulas (e.g., equations) should be written in the
so-called *displayed* math environment:

$$b^{(i)}_k = \begin{cases}
    1 & \text{if 
    $k = \text{argmin}_j \| \mathbf{x}^{(i)} - \mathbf{\mu}_j \|,$}\\
    0 & \text{otherwise}
    \end{cases}$$

Math expressions which you reference in the text should be written
inside the *equation* environment:

$$\label{eq:kmeans-error}
    J = \sum_{i=1}^N \sum_{k=1}^K 
    b^{(i)}_k \| \mathbf{x}^{(i)} - \mathbf{\mu}_k \|^2$$

Now you can reference equation . If the paragraph continues right after
the formula

$$f(x) = x^2 + \varepsilon$$

like this one does, use the command *noindent* after the equation to
remove the indentation of the row.

Multi-letter words in the math environment should be written inside the
command *mathit*, otherwise LaTeX will insert spacing between the
letters to denote the multiplication of values denoted by symbols. For
example, compare $\mathit{Consistent}(h,\mathcal{D})$ and\
$Consistent(h,\mathcal{D})$.

If you need a math symbol, but you don’t know the corresponding LaTeX
command that generates it, try *Detexify*.[^7]

Referencing literature
======================

References to other publications should be written in brackets with the
last name of the first author and the year of publication, e.g.,
[@chomsky-73]. Multiple references are written in sequence, one after
another, separated by semicolon and without whitespaces in between,
e.g., [@chomsky-73; @chave-64; @feigl-58]. References are typically
written at the end of the sentence and necessarily before the sentence
punctuation.

@elkahky-etal-2018-challenge

If the publication is authored by more than one author, only the name of
the first author is written, after which abbreviation *et al.*, meaning
*et alia*, i.e., and others is written as in [@johnson-etc]. If the
publication is authored by only two authors, then the last names of both
authors are written [@johnson-howells].

If the name of the author is incorporated into the text of the sentence,
it should not be in the brackets (only the year should be there).
E.g., “@chomsky-73 suggested that …”. The difference is whether you
reference the publication or the author who wrote it.

The list of all literature references is given alphabetically at the end
of the paper. The form of the reference depends on the type of the
bibliographic unit: conference papers, [@chave-64], books [@butcher-81],
journal articles [@howells-51], doctoral dissertations [@croft-78], and
book chapters [@feigl-58].

All of this is automatically produced when using BibTeX. Insert all the
BibTeX entries into the file `tar2021.bib`, and then reference them via
their symbolic names.

Acknowledgements {#acknowledgements .unnumbered}
================

If suitable, you can include the *Acknowledgements* section before
inserting the literature references in order to thank those who helped
you in any way to deliver the paper, but are not co-authors of the
paper.

[^1]: `https://bit.ly/2SXXn7q`

[^2]: `https://bit.ly/3j1mUau`

[^3]: Some columns were omitted from this preview for clarity

[^4]: `https://nlpaug.readthedocs.io/en/latest/`

[^5]: `https://wordnet.princeton.edu/`

[^6]: `https://www.nltk.org/`

[^7]: `http://detexify.kirelabs.org/`
