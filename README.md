# FLO - CLTV Prediction

![project](/images/project.png)

# Business Problem

[FLO](https://www.flo.com.tr/) is a large e-commerce site in Turkey.

FLO wants to determine a roadmap for its sales and marketing activities.

In order to plan for the medium to long term, the company needs to estimate the potential value that existing customers will provide to the company in the future.

---

# Dataset Info

**Total Features:** 12

**Total Row:** 19.945

| Feature | Definition |
| --- | --- |
| master_id | Unique Customer Number |
| order_channel | Which channel of the shopping platform is used (Android, IOS, Desktop, Mobile) |
| last_order_channel | The channel where the most recent purchase was made |
| first_order_date | Date of the customer's first purchase |
| last_order_channel | Customer's previous shopping history |
| last_order_date_offline | The date of the last purchase made by the customer on the offline platform |
| order_num_total_ever_online | Total number of purchases made by the customer on the online platform |
| order_num_total_ever_offline | Total number of purchases made by the customer on the offline platform |
| customer_value_total_ever_offline | Total fees paid for the customer's offline purchases |
| customer_value_total_ever_online | Total fees paid for the customer's online purchases |
| interested_in_categories_12 | List of categories the customer has shopped in the last 12 months |

---

# Requirements

```python
Lifetimes==0.11.3
pandas==1.4.3
scikit_learn==1.1.2
```

---
# Files

[*FLO_CLTV_Prediction.ipynb*](https://github.com/oguzerdo/flo-cltv-prediction/blob/main/FLO_CLTV_Prediction.ipynb) - CLTV with BG/NBD & GammaGamma Notebook

[*app.py*](https://github.com/oguzerdo/flo-cltv-prediction/blob/main/app.py) - CLTV with BG/NBD & GammaGamma Python Script

---
# Author

[Oğuz Erdoğan](http://www.oguzerdogan.com)
