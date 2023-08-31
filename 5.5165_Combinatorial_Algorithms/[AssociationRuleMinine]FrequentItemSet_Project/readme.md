# Explore Frequent Itemset Generation methods for Association Rule Mining

## Abstract
**Association Rule Mining** is a technique to extract interesting patterns in a large
database. A typical application of association rule mining is **market basket analysis**
where relationships between commodities that tend to be bought together can be
found in a large transaction database. Association Rule Mining requires **generating
all frequent itemset from the database**. However, the task is intrinsic hard because
of both the exponential number of subsets to be considered and the **I/O cost**
of scanning the database. In the project, we study and implement three effective
solutions to the frequent itemset generation problem, **Apriori**[1], **FP-Growth**[5], and
**ECLAT**[14]. We apply these algorithms to two different data set and compare their
efficiency. The result demonstrates that FP-Growth and ECLAT are comparably
efficient while Apriori is much slower


## In `/src` there are
- implementation of:
  - ECLAT
  - FP-Growth
  -  Apriori
- experimental codes

### Detail of methods, setups, and discussions in `FISgen_report.pdf`