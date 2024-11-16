#### The implementation of paper [MMAR: Multilingual and Multimodal Anaphora Resolution in Instructional Videos](https://aclanthology.org/2024.findings-emnlp.88/).

Multilingual anaphora resolution identifies referring expressions and implicit arguments in texts and links to antecedents that cover several languages. In the most challenging setting, cross-lingual anaphora resolution, training data, and test data are in different languages. As knowledge needs to be transferred across languages, this task is challenging, both in the multilingual and cross-lingual setting. We hypothesize that one way to alleviate some of the task's difficulty is to include multimodal information in the form of images (i.e. frames extracted from instructional videos). Such visual inputs are naturally language agnostic, therefore cross- and multilingual anaphora resolution should benefit from visual information. In this paper, we provide the first multilingual and multimodal dataset annotated with anaphoric relations and present experimental results for end-to-end multimodal and multilingual anaphora resolution. Given gold mentions, multimodal features improve anaphora resolution results by ~10 % for unseen languages.


```
pip install torch 
pip install boltons
```

```
python run.py
```
