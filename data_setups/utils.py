import unicodedata
# remove weird text accents (ex: Ã -> A)
def remove_accents(input_str):
    try:
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        clean = ''.join([c for c in nfkd_form if not unicodedata.combining(c)])
    except:
        return "<unk>"
    return clean