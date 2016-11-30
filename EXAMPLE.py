#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import

from tfidf.corpus import Corpus

en_corpus = Corpus(gramsize=3, language='english')
en_corpus['o1'] = """The onion (Allium cepa L., from Latin cepa "onion"), also known as the bulb onion or common onion, is a vegetable and is the most widely cultivated species of the genus Allium."""
en_corpus['o2'] = """This genus also contains several other species variously referred to as onions and cultivated for food, such as the Japanese bunching onion (Allium fistulosum), the tree onion (A. xproliferum), and the Canada onion (Allium canadense). The name "wild onion" is applied to a number of Allium species, but A. cepa is exclusively known from cultivation. Its ancestral wild original form is not known, although escapes from cultivation have become established in some regions.[2] The onion is most frequently a biennial or a perennial plant, but is usually treated as an annual and harvested in its first growing season."""
en_corpus['o3'] = """The onion plant has a fan of hollow, bluish-green leaves and its bulb at the base of the plant begins to swell when a certain day-length is reached. In the autumn (or in spring, in the case of overwintering onions), the foliage dies down and the outer layers of the bulb become dry and brittle. The crop is harvested and dried and the onions are ready for use or storage. The crop is prone to attack by a number of pests and diseases, particularly the onion fly, the onion eelworm, and various fungi cause rotting. Some varieties of A. cepa, such as shallots and potato onions, produce multiple bulbs."""

# Extract the top 50 keywords from the 'o1' document
print(en_corpus.get_keywords(document_id='o1', tf_weight='binary', idf_weight='smooth', limit=50))

# Extract keywords from text you pass in, using the already made corpus
print(en_corpus.get_keywords(text='contains several other species variously', limit=5))

# This example is from the Russian wikipedia page about onions
ru_corpus = Corpus(language='russian')
ru_corpus['o1'] = u"""Лук (лат. Állium) — род двулетних и многолетних травянистых растений, относимых к подсемейству Луковые (Alliaceae) семейства Амариллисовые (Amaryllidaceae) порядка Спаржецветные (Asparagales) (ранее относили к лилейным)."""
ru_corpus['o2'] = u"""Научное латинское название, данное Карлом Линнеем, — лат. allium — происходит от латинского названия чеснока, которое, в свою очередь, вероятно, связано с кельтским словом all — жгучий; другая версия — происходит от лат. halare — пахнуть."""
ru_corpus['o3'] = u"""В роду, по данным сайта The Plant List, более 900 видов[3], которые естественно произрастают в Северном полушарии. Представители рода растут на лугах, в степи, лесах."""

# Extract top 5 keywords
print(ru_corpus.get_keywords(document_id='o2'))