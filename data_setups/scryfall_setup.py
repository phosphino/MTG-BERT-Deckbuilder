import json
import os, sys, re

from sqlalchemy import (MetaData, Table, Column, Integer, create_engine)
import polars as pl
import utils

scryfall_legalities = ['standard', 'pioneer', 'modern', 'explorer', 'historic', 'vintage', 'legacy']

scryfall_feature_columns = 'mana_cost', 'type_line', 'power', 'toughness', 'oracle_text',\
                           'colors', 'name', 'cmc', 'color_identity', 'keywords', 'standard', 'historic', 'arena'

scryfall_path = r"C:\Users\breuh\OneDrive\proggy\python\MTG\roberta\scryfall_json\oracle-cards-20250623212748.json"
scryfall_path = os.path.normpath(scryfall_path)

with open(scryfall_path, 'r', encoding="utf-8") as f:
    scryfall_txt = json.load(f)

for index, card in enumerate(scryfall_txt):
    new_card = dict()
    for k, v in card.items():
        if k == 'name':
            new_card[k] = utils.remove_accents(v)
            continue
        elif k == 'oracle_text':
            new_card[k] = utils.remove_accents(v)
            continue
        new_card[k] = v 
    scryfall_txt[index] = new_card

legality_filter = pl.any_horizontal([
    pl.col("legalities").struct.field(fmt).eq("legal")     # TRUE/FALSE per format
    for fmt in scryfall_legalities
])
df_utf8 = pl.from_dicts(scryfall_txt).with_columns(legality_filter.alias("is_legal")).filter((pl.col("lang") == "en") & pl.col("is_legal"))
df_utf8 = df_utf8.with_columns((df_utf8.select(pl.col('legalities').struct.field('standard')) == 'legal'))
df_utf8 = df_utf8.with_columns((df_utf8.select(pl.col('legalities').struct.field('historic')) == 'legal'))

df_utf8 = df_utf8.with_columns(contains = pl.col('games').list.contains('arena'))

df_utf8 = df_utf8.rename({'contains':'arena'})
df_utf8 = df_utf8.select(scryfall_feature_columns).filter(~pl.col('name').str.starts_with("A-")).sort(by='name')

df_utf8 = df_utf8.unique(subset='name')
list_cols = df_utf8.select(pl.col(pl.List(pl.Utf8))).columns

df_utf8 = df_utf8.with_columns([                              # ➋ build one expression per column
    pl.col(c).list.join(", ").alias(c)              #    now a single-column expr
    for c in list_cols
])

df_for_export = df_utf8.filter(~pl.any_horizontal([pl.col(c) == "<unk>" for c in ['name', 'oracle_text']]))
export_path = os.path.normpath(r"C:\Users\breuh\OneDrive\proggy\python\MTG\roberta\data_setups\training_database.db")
engine_path = 'sqlite:///'+export_path
engine = create_engine(engine_path)

df_for_export.write_database("scryfall_pruned", engine_path, engine='sqlalchemy', if_exists='replace')

text_expr = pl.format(
    "name: {}; type line: {}; keywords: {}; color identity: {}; colors: {}; "
    "mana value: {}; mana cost: {}; power: {}; toughness: {}; text box: {}",
    pl.col("name").fill_null("N/A"),
    pl.col("type_line").fill_null("N/A"),
    pl.col("keywords").fill_null("N/A"),
    pl.col("color_identity").fill_null("N/A"),
    pl.col("colors").fill_null("N/A"),
    pl.col("cmc").fill_null("N/A"),
    pl.col("mana_cost").fill_null("N/A"),
    pl.col("power").fill_null("N/A"),
    pl.col("toughness").fill_null("N/A"),
    pl.col("oracle_text").fill_null("N/A"),
).alias('text')

scryfall_finetuning = df_for_export.select(text_expr)["text"].to_frame()
scryfall_finetuning = pl.concat([df_for_export.select(pl.col('name')), scryfall_finetuning], how='horizontal').sort(by='name')

scryfall_finetuning.write_database("scryfall_finetuning", engine_path, engine='sqlalchemy', if_exists='replace')