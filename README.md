# Additional Experiment applying RLMRec on the Katchers dataset


This repository is a branch that covers additional experiments for the paper "Enhancing e-commerce recommendation systems with multiple item purchase data: a Bidirectional Encoder Representations from Transformers-based approach". Specifically, it deals with the procedure of applying the main dataset of the paper, the Katchers dataset, to RLMRec, which is the latest state-of-the-art (SOTA) model.

<img src='RLMRec_cover.png' />

 
 >**Representation Learning with Large Language Models for Recommendation**  
 >Xubin Ren, Wei Wei, Lianghao Xia, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, Chao Huang*\
 >*WWW2024*

For smooth experimentation, I forked the code from the RLMRec original author's repository at https://github.com/HKUDS/RLMRec.git, created this branch named `dev_katchers` and conducted my work there.





## Setup
Clone the following branch to your local machine:
```bash
git clone https://github.com/jangmino/RLMRec/tree/dev-katchers
```

If necessary, follow the procedures in the original author's repository to install the required packages.
In my case, the code ran successfully with Python 3.10 and PyTorch 2.3 with CUDA 12.2.

## Download Processed Data

Please download [[Google Drive](https://drive.google.com/file/d/1n2xQDz4vWZ4pDghqJcBvsCI7mrsVwVLb/view?usp=sharing)], the tar-compressed file, `katchers_rlmrec_data.tar.gz`. And uncompress at `data/katchers`.

The dataset was created with the same folder structure and format as the original author's amazon data. 
```
- katchers
|--- trn_mat.pkl    # training set (sparse matrix)
|--- val_mat.pkl    # validation set (sparse matrix)
|--- tst_mat.pkl    # test set (sparse matrix)
|--- usr_prf.pkl    # text description of users
|--- itm_prf.pkl    # text description of items
|--- usr_emb_np.pkl # user text embeddings
|--- itm_emb_np.pkl # item text embeddings
```

One additional file is `tst_masked_unquery_iids.pkl`. This is auxiliary information necessary to conduct an identical evaluation as in our paper's experiments. In our test set construction, we grouped one positive example with 49 negative examples for prediction per each user, followed by measuring ndcg5 and ndcg10. This file is used to mask all products except for the 50 (1+49) items per user in the test set.


## Examples to run the train and get the test metrics.

  - RLMRec-Con **(Constrastive Alignment)**:

    ```python encoder/train_encoder.py --model {model_name}_plus --dataset katchers --cuda 0```

  - RLMRec-Gen **(Generative Alignment)**:

    ```python encoder/train_encoder.py --model {model_name}_gene --dataset katchers--cuda 0```


## Test Results

| Model | NDCG@5 | NDCG@10 |
| --- | --- | --- |
|BERT With|0.4910*|0.5376*|
|gccf_plus|0.3100|0.3616|
|gccf_gene|0.3022|0.3512|
|dccf_plus|0.2894|0.3374|
|dccf_gene|0.2387|0.2836|
|lightgcn_plus|0.3132|0.3623|
|lightgcn_gene|0.2942|0.3411|

The results of applying RLMRec show lower performance compared to the method proposed in our paper. However, this does not necessarily mean that our proposed method overwhelmingly outperforms RLMRec. The reasons are as follows:
- In the RLMRec study, datasets like Amazon could utilize information such as item descriptions and reviews as materials for item and user profiles. However, the Katchers dataset lacks this information, so we could only use the product titles.
- RLMRec could not model multiple item purchase information.

In conclusion, it is difficult to assert which method is superior because the proposed method and RLMRec use different materials and contexts.


# Processing Steps for building Necessrafy Files 

In the `notebook`` folder, you can find the following code:

`build_async_item_profile_data.ipynb`
- Generates request data in JSON format for OpenAI async API calls to create item profiles
- Interprets OpenAI API response results and transforms them into item profiles
- Generates request data in JSON format for OpenAI API calls to obtain item embeddings
- Interprets response results and transforms them into item embeddings

`build_async_user_profile_data.ipynb`
- Generates request data in JSON format for OpenAI async API calls to create user profiles
- Interprets OpenAI API response results and transforms them into user profiles
- Generates request data in JSON format for OpenAI API calls to obtain user embeddings
- Interprets response results and transforms them into user embeddings

`convert-katchers-data.ipynb`
- Processes data to create coo_matrix
- Generates other necessary pkl files

