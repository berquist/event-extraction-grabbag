from flexnlp.model.doc_test_utils import TestDoc


doc = TestDoc("Here are some tokens for testing").to_doc()
print(doc)
tokens = doc.tokens()
here, are, some, _, _, testing = tokens
print(here)
print(are)
print(some)
print(testing)

assert testing.index - here.index == 5
assert some.index - here.index == 2
