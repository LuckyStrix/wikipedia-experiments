"""Inline templates rendered as the text a reader sees (wikiexp/wikitemplates.py), and the dangling
punctuation that used to be left where a template vanished."""
import re

import pytest

from wikiexp import wikitemplates as T
from wikiexp import wikitext as W


@pytest.mark.parametrize("wikitext,expected", [
    # convert / cvt: value, unit and (when cheap) the conversion
    ("a listed height of {{convert|300|m}}, tall", "a listed height of 300 m (980 ft), tall"),
    ("{{cvt|300|m|ft|abbr=on}}", "300 m (980 ft)"),
    ("{{convert|10|km|mi}}", "10 km (6.2 mi)"),
    ("{{convert|5|mi|km|0}}", "5 mi (8 km)"),
    ("{{convert|5|mi|km|1}}", "5 mi (8.0 km)"),
    ("{{convert|100|km/h|mph}}", "100 km/h (62 mph)"),
    ("{{convert|20|C}} and {{convert|68|°F|°C}}", "20 °C (68 °F) and 68 °F (20 °C)"),
    ("{{convert|10|to|20|km}}", "10 to 20 km (6.2 to 12 mi)"),
    ("{{convert|7|-|10|in|mm}}", "7–10 in (180–250 mm)"),
    ("{{cvt|31 and 33|°C}}", "31 and 33 °C (88 and 91 °F)"),
    ("{{convert|5|ft|6|in|m}}", "5 ft 6 in (1.7 m)"),
    ("{{convert|1.8|m|ftin}}", "1.8 m (5 ft 11 in)"),
    ("{{convert|5|e6km2|e6sqmi}}", "5 million km² (1.9 million sq mi)"),
    ("{{convert|300|sqft|disp=flip}}", "28 m² (300 sq ft)"),
    ("{{convert|300|m|disp=out}}", "980 ft"),
    ("{{convert|1|acre}}", "1 acre (0.40 ha)"),
    ("{{convert|40|bhp}}", "40 bhp"),                      # unknown unit: as written, never nothing
    ("{{convert|5/8|hp}}", "5/8 hp"),
    ("{{convert|1,000|km}}", "1,000 km (620 mi)"),
    ("{{convert|12|kg}}", "12 kg (26 lb)"),
    # dates
    ("born {{birth date and age|1950|3|7}}", "born March 7, 1950"),
    ("{{birth date|1950|03|07|df=yes}}", "7 March 1950"),
    ("{{death date and age|1973|03|11|1904|05|06|df=y}}", "11 March 1973"),
    ("{{start date|1994}}", "1994"),
    ("{{end date|2005|9}}", "September 2005"),
    ("{{date|2023-05-09|mdy}} {{date|2023-05-09|dmy}}", "May 9, 2023 9 May 2023"),
    ("{{as of|2020}}, {{as of|2020|lc=y}}, {{As of|2020|5}}, {{as of|2020|5|3}}",
     "As of 2020, as of 2020, As of May 2020, As of 3 May 2020"),
    ("{{as of|2020|alt=in 2020}}", "in 2020"),
    ("{{circa|1500}} {{c.|1500|1510}}", "c. 1500 c. 1500–1510"),
    ("{{reign|336|323|era=BC}}", "r. 336 – 323 BC"),
    ("{{BCE|563–483}} {{CE|9}}", "563–483 BCE 9 CE"),
    ("{{marriage|Ida Dalser|1914|1915|end=div}}", "Ida Dalser (m. 1914; div. 1915)"),
    # languages and names
    ("{{lang|fr|bonjour}} {{lang-de|Hallo}} {{langx|es|hola}}", "bonjour Hallo hola"),
    ("{{transl|ru|ALA-LC|Moskva}}", "Moskva"),
    ("{{nihongo|Tokyo|東京|Tōkyō}}", "Tokyo (東京, Tōkyō)"),
    ("{{nihongo||東京|Tōkyō}}", "東京, Tōkyō"),
    ("{{nihongo|Cat|猫}}{{nihongo|x|y|z|post=,}}", "Cat (猫)x (y, z),"),
    ("{{zh|c=北京|p=Běijīng}} {{zh|labels=no|c=北京|p=Běijīng}}", "Chinese: 北京; pinyin: Běijīng 北京; Běijīng"),
    ("{{script|Arab|مرحبا}}", "مرحبا"),
    ("{{gloss|dawn}} {{lit|the sea}}", "‘dawn’ lit. the sea"),
    ("{{nee|Darragh}}", "née Darragh"),
    # typography and small things
    ("{{nowrap|a b}} {{abbr|NASA|National Aeronautics}} {{sic|teh}} {{small|x}}", "a b NASA teh x"),
    ("{{frac|1|2}} {{frac|3}} {{frac|5|1|2}}", "1/2 1/3 5 1/2"),
    ("{{ordinal|1}} {{ordinal|12}} {{ordinal|23}} {{ordinal|104}}", "1st 12th 23rd 104th"),
    ("{{val|1.5|e=6|u=km}} {{val|1234567}}", "1.5×10^6 km 1,234,567"),
    ("{{formatnum:1234567}} {{#expr:1000-1}}", "1,234,567 999"),
    ("{{angbr|j}} {{e|13}} {{space|2}}x", "⟨j⟩ ×10^13 x"),
    ("{{ISBN|0-415-01035-7}}", "ISBN 0-415-01035-7"),
    ("{{bibleverse|Matthew|5:3–10|ESV}}", "Matthew 5:3–10"),
    ("{{USS|Essex|CV-9|6}} {{HMS|Beagle}} {{sclass|Acciaio|submarine|4}} {{ship|Italian battleship|Roma||2}}",
     "USS Essex HMS Beagle Acciaio-class submarine Italian battleship Roma"),
    ("{{chem|UO|2|F|2}} {{chem2|NF3}} {{chem2|[Fe(CN)6]^3-}} {{nuclide|uranium|238}}",
     "UO2F2 NF3 [Fe(CN)6]3- uranium-238"),
    ("{{CAD|145,000}} {{US$|5}} {{£|3}}", "C$145,000 US$5 £3"),
    ("{{stnlnk|Truro}} and {{stnlink|Penryn}}", "Truro and Penryn"),
])
def test_rendered_templates(wikitext, expected):
    assert W.clean(wikitext) == expected


@pytest.mark.parametrize("wikitext", [
    "{{cite book|title=X}}", "{{sfn|Smith|2000|p=5}}", "{{citation needed|date=May 2020}}",
    "{{efn|a note}}", "{{Infobox person|name=X}}", "{{flagicon|Hungary}}", "{{rp|81}}", "{{r|abc}}",
    "{{main|Paris}}", "{{Unknown template|x|y}}", "{{IPA|en|ˈfoʊ}}", "{{IPAc-en|ˈ|f|oʊ}}", "{{respell|FOH}}",
    "{{audio|x.ogg|Play}}", "{{pronunciation|x.ogg}}",
])
def test_templates_that_carry_no_prose_still_vanish(wikitext):
    assert W.clean(wikitext) == ""


@pytest.mark.parametrize("wikitext,expected", [
    ("Rouen ({{IPAc-en|UK|r|uː}}, {{IPAc-en|US|r}};{{cite x}} {{IPA|fr|ʁwɑ̃}} <small>or</small> {{IPA|fr|x}}) is a city",
     "Rouen is a city"),
    ("Giuliani ({{IPA-en|x}}; born May 28, 1944) is", "Giuliani (born May 28, 1944) is"),
    ("Dakar ({{IPA|wo|x}}) is", "Dakar is"),
    ("a ( , ; ) b", "a b"),
    ("it was {{convert||}} here", "it was here"),
    ("text{{sfn|x}}, more", "text, more"),
])
def test_no_dangling_punctuation_where_a_template_vanished(wikitext, expected):
    assert W.clean(wikitext) == expected


def test_render_returns_none_for_unknown_and_never_raises():
    assert T.render("sfn", ["a"], {}) is None
    assert T.render("convert", [], {}) == ""
    assert T.render("convert", ["x", "y", "z"], {}) == "x y"
    assert T.render("frac", [], {}) == ""
    assert T.render("lang-fr", ["bonjour"], {}) == "bonjour"
    assert T.render("ipa-fr", ["x"], {}) == ""
    assert T.render("#expr:1/0", [], {}) == ""
    assert T.render("#expr:__import__('os')", [], {}) is None      # never evaluates arbitrary code
    assert T.render("#expr:2**3", [], {}) == ""


def test_sig_figs_and_formatting():
    assert [T._sig_figs(s) for s in ("300", "1,000", "1.50", "250", "0.0042")] == [1, 1, 3, 2, 2]
    assert T._fmt(984.25, 2) == "980" and T._fmt(3.1069, 2) == "3.1" and T._fmt(0, 2) == "0"
    assert T._fmt(12345.6, 3) == "12,300" and T._fmt(0.04567, 2) == "0.046"


def test_unit_conversions_are_accurate():
    assert W.clean("{{convert|1|mi|km|3}}") == "1 mi (1.609 km)"
    assert W.clean("{{convert|0|C|F}}") == "0 °C (32 °F)"
    assert W.clean("{{convert|100|C|K|0}}") == "100 °C (373 K)"
    assert W.clean("{{convert|1|lb|kg|3}}") == "1 lb (0.454 kg)"
    assert W.clean("{{convert|1|nmi|km|3}}") == "1 nmi (1.852 km)"


def test_known_leftover_patterns_are_gone_from_a_realistic_paragraph():
    wikitext = ("The tower has a listed height of {{convert|300|m|abbr=on}}, a pinnacle of {{cvt|24|m|0}} and an "
                "area of {{convert|3.5|ha}} ({{IPAc-en|ˈ|t|aʊ|ər}}). It opened on {{start date|1889|3|31|df=y}}; "
                "{{As of|2020|lc=y}} it has {{formatnum:7000000}} visitors a year.{{sfn|Smith|2000|p=5}}")
    out = W.clean(wikitext)
    assert out == ("The tower has a listed height of 300 m (980 ft), a pinnacle of 24 m (79 ft) and an area of "
                   "3.5 ha (8.6 acres). It opened on 31 March 1889; as of 2020 it has 7,000,000 visitors a year.")
    assert not re.search(r"of,|\(\s*[,;]|[,;]\s*\)|\s,", out)
