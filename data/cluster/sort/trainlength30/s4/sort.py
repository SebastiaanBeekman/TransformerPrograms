import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys] for q in queries]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/length/rasp/sort/trainlength30/s4/sort_weights.csv",
        index_col=[0, 1],
        dtype={"feature": str},
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(q_token, k_token):
        if q_token in {"1", "0"}:
            return k_token == "0"
        elif q_token in {"2", "</s>"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_0_0_pattern = select_closest(tokens, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(position, token):
        if position in {0, 11, 13, 22, 29}:
            return token == "1"
        elif position in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 24}:
            return token == "2"
        elif position in {12, 16, 18, 19, 21}:
            return token == "4"
        elif position in {27, 23, 14, 15}:
            return token == "0"
        elif position in {32, 33, 36, 37, 38, 39, 17, 30, 31}:
            return token == ""
        elif position in {25}:
            return token == "<s>"
        elif position in {26}:
            return token == "</s>"
        elif position in {28}:
            return token == "3"
        elif position in {34, 35}:
            return token == "<pad>"

    attn_0_1_pattern = select_closest(tokens, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 16, 19}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2, 11}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {27, 4, 5, 15}:
            return k_position == 6
        elif q_position in {10, 6}:
            return k_position == 7
        elif q_position in {33, 7}:
            return k_position == 11
        elif q_position in {8}:
            return k_position == 12
        elif q_position in {9}:
            return k_position == 13
        elif q_position in {12, 29, 14}:
            return k_position == 9
        elif q_position in {26, 13}:
            return k_position == 8
        elif q_position in {17}:
            return k_position == 10
        elif q_position in {18}:
            return k_position == 20
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 22
        elif q_position in {32, 22, 39}:
            return k_position == 23
        elif q_position in {23}:
            return k_position == 24
        elif q_position in {24}:
            return k_position == 25
        elif q_position in {25, 35}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 0
        elif q_position in {30}:
            return k_position == 28
        elif q_position in {31}:
            return k_position == 33
        elif q_position in {34, 38}:
            return k_position == 39
        elif q_position in {36}:
            return k_position == 38
        elif q_position in {37}:
            return k_position == 5

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 3, 38, 20, 25}:
            return k_position == 5
        elif q_position in {1, 10, 14}:
            return k_position == 4
        elif q_position in {2, 28, 7}:
            return k_position == 3
        elif q_position in {9, 4, 30}:
            return k_position == 8
        elif q_position in {29, 5}:
            return k_position == 7
        elif q_position in {6, 16, 17, 23, 27}:
            return k_position == 14
        elif q_position in {8, 12, 15}:
            return k_position == 11
        elif q_position in {11, 37}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 9
        elif q_position in {18}:
            return k_position == 12
        elif q_position in {19}:
            return k_position == 16
        elif q_position in {21}:
            return k_position == 19
        elif q_position in {22}:
            return k_position == 28
        elif q_position in {24, 26}:
            return k_position == 25
        elif q_position in {34, 31}:
            return k_position == 34
        elif q_position in {32}:
            return k_position == 24
        elif q_position in {33}:
            return k_position == 10
        elif q_position in {35}:
            return k_position == 27
        elif q_position in {36}:
            return k_position == 39
        elif q_position in {39}:
            return k_position == 38

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 8, 4, 9}:
            return k_position == 5
        elif q_position in {1, 27, 13, 15}:
            return k_position == 10
        elif q_position in {2}:
            return k_position == 16
        elif q_position in {3}:
            return k_position == 1
        elif q_position in {28, 5}:
            return k_position == 15
        elif q_position in {11, 29, 6, 7}:
            return k_position == 4
        elif q_position in {10, 26, 14}:
            return k_position == 8
        elif q_position in {12}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 12
        elif q_position in {17, 18}:
            return k_position == 13
        elif q_position in {19, 20}:
            return k_position == 18
        elif q_position in {21, 22}:
            return k_position == 20
        elif q_position in {23}:
            return k_position == 26
        elif q_position in {24, 31}:
            return k_position == 19
        elif q_position in {25}:
            return k_position == 17
        elif q_position in {30}:
            return k_position == 32
        elif q_position in {32, 39}:
            return k_position == 28
        elif q_position in {33, 35}:
            return k_position == 34
        elif q_position in {34}:
            return k_position == 33
        elif q_position in {36}:
            return k_position == 38
        elif q_position in {37}:
            return k_position == 29
        elif q_position in {38}:
            return k_position == 31

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(position, token):
        if position in {0, 17, 12, 15}:
            return token == "1"
        elif position in {1, 37, 16, 18, 26}:
            return token == "0"
        elif position in {19, 2, 10, 14}:
            return token == "4"
        elif position in {33, 3, 35, 36, 38, 39, 9, 31}:
            return token == ""
        elif position in {4, 5, 6, 7, 8, 21, 24}:
            return token == "2"
        elif position in {11, 13, 20, 22, 23, 27, 28}:
            return token == "3"
        elif position in {32, 34, 25, 29, 30}:
            return token == "</s>"

    attn_0_5_pattern = select_closest(tokens, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(position, token):
        if position in {0, 38, 14, 22}:
            return token == "1"
        elif position in {1, 2, 4}:
            return token == "0"
        elif position in {3, 5, 6, 7, 8, 9, 12, 21, 28}:
            return token == "3"
        elif position in {10, 15, 18, 20, 29}:
            return token == "2"
        elif position in {11, 13, 16, 17, 19, 26, 27}:
            return token == "4"
        elif position in {25, 23}:
            return token == "</s>"
        elif position in {24}:
            return token == "<pad>"
        elif position in {32, 33, 34, 35, 36, 37, 39, 30, 31}:
            return token == ""

    attn_0_6_pattern = select_closest(tokens, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(position, token):
        if position in {0, 19, 21, 22}:
            return token == "3"
        elif position in {1, 2, 33, 36, 16, 18}:
            return token == "0"
        elif position in {17, 3, 15}:
            return token == "1"
        elif position in {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20, 25}:
            return token == "4"
        elif position in {24, 28, 29, 23}:
            return token == "2"
        elif position in {26}:
            return token == "</s>"
        elif position in {27}:
            return token == "<s>"
        elif position in {32, 34, 35, 37, 38, 39, 30, 31}:
            return token == ""

    attn_0_7_pattern = select_closest(tokens, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 28}:
            return token == "2"
        elif position in {1, 2, 29}:
            return token == "1"
        elif position in {32, 33, 3, 4, 5, 35, 36, 37, 38, 39, 27, 31}:
            return token == "0"
        elif position in {34, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 30}:
            return token == ""
        elif position in {18}:
            return token == "<pad>"
        elif position in {25, 26, 20}:
            return token == "<s>"
        elif position in {24, 21, 22, 23}:
            return token == "4"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {
            0,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            34,
            36,
            37,
        }:
            return token == ""
        elif position in {1}:
            return token == "<s>"
        elif position in {33, 2, 3, 4, 5, 6, 7, 8, 9, 10, 38, 30}:
            return token == "1"
        elif position in {29, 11, 12, 13}:
            return token == "0"
        elif position in {32, 35, 39}:
            return token == "2"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0, 38, 39, 8, 30}:
            return k_position == 14
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 8
        elif q_position in {3, 29}:
            return k_position == 9
        elif q_position in {4}:
            return k_position == 10
        elif q_position in {5}:
            return k_position == 11
        elif q_position in {6, 7}:
            return k_position == 13
        elif q_position in {9, 35}:
            return k_position == 16
        elif q_position in {10, 36}:
            return k_position == 17
        elif q_position in {11, 12}:
            return k_position == 20
        elif q_position in {13, 14}:
            return k_position == 21
        elif q_position in {15}:
            return k_position == 23
        elif q_position in {16, 17, 18}:
            return k_position == 26
        elif q_position in {19, 21}:
            return k_position == 28
        elif q_position in {20, 23}:
            return k_position == 32
        elif q_position in {22}:
            return k_position == 35
        elif q_position in {24, 26}:
            return k_position == 30
        elif q_position in {25, 27}:
            return k_position == 34
        elif q_position in {28}:
            return k_position == 36
        elif q_position in {32, 34, 31}:
            return k_position == 15
        elif q_position in {33}:
            return k_position == 24
        elif q_position in {37}:
            return k_position == 19

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0, 10}:
            return k_position == 14
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {3, 4}:
            return k_position == 7
        elif q_position in {33, 5, 30}:
            return k_position == 9
        elif q_position in {37, 6}:
            return k_position == 10
        elif q_position in {7}:
            return k_position == 11
        elif q_position in {8}:
            return k_position == 12
        elif q_position in {9}:
            return k_position == 13
        elif q_position in {11}:
            return k_position == 16
        elif q_position in {12}:
            return k_position == 18
        elif q_position in {13, 14}:
            return k_position == 19
        elif q_position in {38, 15}:
            return k_position == 21
        elif q_position in {16, 39}:
            return k_position == 22
        elif q_position in {17}:
            return k_position == 23
        elif q_position in {18, 29}:
            return k_position == 24
        elif q_position in {19}:
            return k_position == 26
        elif q_position in {20}:
            return k_position == 25
        elif q_position in {25, 35, 21, 22}:
            return k_position == 27
        elif q_position in {23}:
            return k_position == 29
        elif q_position in {24}:
            return k_position == 39
        elif q_position in {26, 27}:
            return k_position == 32
        elif q_position in {32, 28}:
            return k_position == 30
        elif q_position in {34, 31}:
            return k_position == 8
        elif q_position in {36}:
            return k_position == 1

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {0, 24, 22}:
            return token == "1"
        elif position in {32, 1, 2, 3, 4, 35, 21, 23, 29, 30}:
            return token == "0"
        elif position in {
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            15,
            16,
            17,
            18,
            19,
            25,
            26,
            31,
            33,
            34,
            36,
            37,
            38,
            39,
        }:
            return token == ""
        elif position in {14}:
            return token == "<pad>"
        elif position in {20}:
            return token == "<s>"
        elif position in {27, 28}:
            return token == "</s>"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(q_position, k_position):
        if q_position in {0, 1, 9}:
            return k_position == 12
        elif q_position in {2, 3, 31}:
            return k_position == 6
        elif q_position in {4, 5, 36}:
            return k_position == 8
        elif q_position in {6}:
            return k_position == 9
        elif q_position in {7}:
            return k_position == 10
        elif q_position in {8}:
            return k_position == 11
        elif q_position in {10}:
            return k_position == 13
        elif q_position in {11}:
            return k_position == 14
        elif q_position in {12, 14}:
            return k_position == 16
        elif q_position in {13, 15}:
            return k_position == 17
        elif q_position in {16}:
            return k_position == 20
        elif q_position in {17}:
            return k_position == 21
        elif q_position in {18}:
            return k_position == 22
        elif q_position in {19}:
            return k_position == 24
        elif q_position in {20}:
            return k_position == 27
        elif q_position in {21}:
            return k_position == 26
        elif q_position in {22, 23}:
            return k_position == 28
        elif q_position in {24, 25, 26, 27}:
            return k_position == 29
        elif q_position in {28}:
            return k_position == 37
        elif q_position in {29}:
            return k_position == 1
        elif q_position in {32, 33, 30, 39}:
            return k_position == 2
        elif q_position in {34, 35, 37}:
            return k_position == 3
        elif q_position in {38}:
            return k_position == 34

    num_attn_0_5_pattern = select(positions, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(q_position, k_position):
        if q_position in {0, 12}:
            return k_position == 14
        elif q_position in {1}:
            return k_position == 17
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {4, 36}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {32, 37, 6, 30}:
            return k_position == 8
        elif q_position in {33, 29, 7}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 10
        elif q_position in {9}:
            return k_position == 11
        elif q_position in {10}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 15
        elif q_position in {14}:
            return k_position == 18
        elif q_position in {16, 17, 15}:
            return k_position == 19
        elif q_position in {18}:
            return k_position == 20
        elif q_position in {19}:
            return k_position == 22
        elif q_position in {20}:
            return k_position == 23
        elif q_position in {21}:
            return k_position == 25
        elif q_position in {22, 23}:
            return k_position == 26
        elif q_position in {24}:
            return k_position == 28
        elif q_position in {25}:
            return k_position == 34
        elif q_position in {26, 27, 28, 38}:
            return k_position == 35
        elif q_position in {35, 39, 31}:
            return k_position == 0
        elif q_position in {34}:
            return k_position == 30

    num_attn_0_6_pattern = select(positions, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {0, 1, 2, 3, 4, 5, 32, 33, 8, 34, 36, 38, 39, 30, 31}:
            return token == ""
        elif position in {37, 12, 29, 6}:
            return token == "</s>"
        elif position in {35, 7}:
            return token == "<pad>"
        elif position in {9, 10, 11, 13}:
            return token == "<s>"
        elif position in {14}:
            return token == "2"
        elif position in {15, 17, 19, 20, 22, 24, 25, 27}:
            return token == "1"
        elif position in {16, 18, 21, 23, 26, 28}:
            return token == "0"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position):
        key = position
        if key in {
            11,
            12,
            13,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            34,
            38,
        }:
            return 20
        elif key in {1}:
            return 11
        elif key in {6}:
            return 18
        return 36

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in positions]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_5_output, position):
        key = (attn_0_5_output, position)
        if key in {
            ("0", 2),
            ("0", 3),
            ("0", 4),
            ("0", 5),
            ("0", 31),
            ("0", 35),
            ("0", 36),
            ("1", 2),
            ("1", 3),
            ("1", 4),
            ("1", 5),
            ("2", 2),
            ("2", 3),
            ("2", 4),
            ("2", 5),
            ("2", 30),
            ("2", 31),
            ("2", 32),
            ("2", 33),
            ("2", 34),
            ("2", 35),
            ("2", 36),
            ("2", 37),
            ("2", 38),
            ("2", 39),
            ("3", 2),
            ("3", 3),
            ("3", 4),
            ("3", 5),
            ("4", 2),
            ("4", 3),
            ("4", 4),
            ("4", 5),
            ("</s>", 2),
            ("</s>", 3),
            ("</s>", 4),
            ("</s>", 5),
            ("<s>", 2),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 5),
            ("<s>", 30),
            ("<s>", 31),
            ("<s>", 32),
            ("<s>", 33),
            ("<s>", 34),
            ("<s>", 35),
            ("<s>", 36),
            ("<s>", 37),
            ("<s>", 38),
            ("<s>", 39),
        }:
            return 35
        elif key in {
            ("0", 29),
            ("1", 1),
            ("1", 29),
            ("2", 1),
            ("2", 29),
            ("3", 1),
            ("3", 29),
            ("4", 1),
            ("4", 29),
            ("</s>", 1),
            ("</s>", 29),
            ("<s>", 1),
            ("<s>", 29),
        }:
            return 30
        elif key in {("0", 1)}:
            return 8
        return 2

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_5_outputs, positions)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(position):
        key = position
        if key in {3, 4, 5, 6, 29}:
            return 30
        elif key in {0, 7}:
            return 32
        return 13

    mlp_0_2_outputs = [mlp_0_2(k0) for k0 in positions]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(position):
        key = position
        if key in {1, 2, 3, 29, 30, 31, 32, 33, 35, 36}:
            return 0
        elif key in {4}:
            return 30
        return 10

    mlp_0_3_outputs = [mlp_0_3(k0) for k0 in positions]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_3_output, num_attn_0_6_output):
        key = (num_attn_0_3_output, num_attn_0_6_output)
        if key in {(0, 0)}:
            return 32
        return 4

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_2_output):
        key = num_attn_0_2_output
        if key in {0}:
            return 27
        return 11

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_0_output):
        key = num_attn_0_0_output
        return 33

    num_mlp_0_2_outputs = [num_mlp_0_2(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_2_output, num_attn_0_3_output):
        key = (num_attn_0_2_output, num_attn_0_3_output)
        return 37

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(mlp_0_2_output, num_mlp_0_0_output):
        if mlp_0_2_output in {0, 1, 32, 36, 39, 8, 23, 29}:
            return num_mlp_0_0_output == 13
        elif mlp_0_2_output in {2}:
            return num_mlp_0_0_output == 15
        elif mlp_0_2_output in {35, 3, 13, 15, 21}:
            return num_mlp_0_0_output == 4
        elif mlp_0_2_output in {33, 34, 4, 5, 9, 19}:
            return num_mlp_0_0_output == 26
        elif mlp_0_2_output in {6}:
            return num_mlp_0_0_output == 17
        elif mlp_0_2_output in {7}:
            return num_mlp_0_0_output == 18
        elif mlp_0_2_output in {24, 10, 11, 12}:
            return num_mlp_0_0_output == 6
        elif mlp_0_2_output in {30, 14, 22}:
            return num_mlp_0_0_output == 32
        elif mlp_0_2_output in {16}:
            return num_mlp_0_0_output == 10
        elif mlp_0_2_output in {17, 18}:
            return num_mlp_0_0_output == 20
        elif mlp_0_2_output in {25, 20, 37}:
            return num_mlp_0_0_output == 5
        elif mlp_0_2_output in {26, 38}:
            return num_mlp_0_0_output == 2
        elif mlp_0_2_output in {27}:
            return num_mlp_0_0_output == 28
        elif mlp_0_2_output in {28}:
            return num_mlp_0_0_output == 11
        elif mlp_0_2_output in {31}:
            return num_mlp_0_0_output == 14

    attn_1_0_pattern = select_closest(
        num_mlp_0_0_outputs, mlp_0_2_outputs, predicate_1_0
    )
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_4_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_3_output, token):
        if attn_0_3_output in {"0"}:
            return token == "2"
        elif attn_0_3_output in {"2", "1", "</s>", "3"}:
            return token == "3"
        elif attn_0_3_output in {"4", "<s>"}:
            return token == "4"

    attn_1_1_pattern = select_closest(tokens, attn_0_3_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(token, mlp_0_3_output):
        if token in {"3", "4", "</s>", "0"}:
            return mlp_0_3_output == 0
        elif token in {"1"}:
            return mlp_0_3_output == 30
        elif token in {"2"}:
            return mlp_0_3_output == 10
        elif token in {"<s>"}:
            return mlp_0_3_output == 24

    attn_1_2_pattern = select_closest(mlp_0_3_outputs, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_2_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(mlp_0_2_output, num_mlp_0_0_output):
        if mlp_0_2_output in {0, 33, 38, 39, 14, 17}:
            return num_mlp_0_0_output == 6
        elif mlp_0_2_output in {1, 11, 13}:
            return num_mlp_0_0_output == 4
        elif mlp_0_2_output in {2, 6, 20, 26, 31}:
            return num_mlp_0_0_output == 5
        elif mlp_0_2_output in {3}:
            return num_mlp_0_0_output == 1
        elif mlp_0_2_output in {32, 4, 22}:
            return num_mlp_0_0_output == 26
        elif mlp_0_2_output in {5, 21, 24, 29, 30}:
            return num_mlp_0_0_output == 3
        elif mlp_0_2_output in {18, 12, 37, 7}:
            return num_mlp_0_0_output == 7
        elif mlp_0_2_output in {8}:
            return num_mlp_0_0_output == 15
        elif mlp_0_2_output in {9, 15}:
            return num_mlp_0_0_output == 12
        elif mlp_0_2_output in {10}:
            return num_mlp_0_0_output == 36
        elif mlp_0_2_output in {16}:
            return num_mlp_0_0_output == 19
        elif mlp_0_2_output in {19, 28}:
            return num_mlp_0_0_output == 14
        elif mlp_0_2_output in {23}:
            return num_mlp_0_0_output == 2
        elif mlp_0_2_output in {25}:
            return num_mlp_0_0_output == 37
        elif mlp_0_2_output in {27}:
            return num_mlp_0_0_output == 17
        elif mlp_0_2_output in {34}:
            return num_mlp_0_0_output == 20
        elif mlp_0_2_output in {35}:
            return num_mlp_0_0_output == 13
        elif mlp_0_2_output in {36}:
            return num_mlp_0_0_output == 21

    attn_1_3_pattern = select_closest(
        num_mlp_0_0_outputs, mlp_0_2_outputs, predicate_1_3
    )
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_0_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(num_mlp_0_3_output, token):
        if num_mlp_0_3_output in {
            0,
            1,
            2,
            3,
            4,
            32,
            33,
            7,
            35,
            39,
            15,
            18,
            24,
            25,
            27,
            30,
            31,
        }:
            return token == ""
        elif num_mlp_0_3_output in {5, 6, 9, 10, 14, 20, 26}:
            return token == "4"
        elif num_mlp_0_3_output in {36, 37, 8, 17, 19, 22, 29}:
            return token == "2"
        elif num_mlp_0_3_output in {34, 38, 11, 12, 13, 21, 23, 28}:
            return token == "3"
        elif num_mlp_0_3_output in {16}:
            return token == "1"

    attn_1_4_pattern = select_closest(tokens, num_mlp_0_3_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, tokens)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(position, attn_0_1_output):
        if position in {0, 25, 36, 23}:
            return attn_0_1_output == "0"
        elif position in {1, 35, 39}:
            return attn_0_1_output == "1"
        elif position in {33, 2, 3, 4, 5, 34, 37, 9, 15, 19, 20, 22, 24, 26, 30, 31}:
            return attn_0_1_output == ""
        elif position in {16, 6}:
            return attn_0_1_output == "<s>"
        elif position in {38, 7, 8, 10, 13, 18}:
            return attn_0_1_output == "4"
        elif position in {32, 11, 12, 14, 17, 27, 29}:
            return attn_0_1_output == "2"
        elif position in {21}:
            return attn_0_1_output == "3"
        elif position in {28}:
            return attn_0_1_output == "</s>"

    attn_1_5_pattern = select_closest(attn_0_1_outputs, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_0_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(mlp_0_0_output, position):
        if mlp_0_0_output in {0}:
            return position == 8
        elif mlp_0_0_output in {24, 1, 2, 6}:
            return position == 13
        elif mlp_0_0_output in {3}:
            return position == 32
        elif mlp_0_0_output in {4, 12, 17, 18, 21}:
            return position == 5
        elif mlp_0_0_output in {5}:
            return position == 7
        elif mlp_0_0_output in {7, 39, 9, 11, 15, 16, 23}:
            return position == 30
        elif mlp_0_0_output in {8}:
            return position == 10
        elif mlp_0_0_output in {10, 13}:
            return position == 29
        elif mlp_0_0_output in {33, 20, 14}:
            return position == 4
        elif mlp_0_0_output in {19}:
            return position == 9
        elif mlp_0_0_output in {34, 22}:
            return position == 15
        elif mlp_0_0_output in {25, 36}:
            return position == 2
        elif mlp_0_0_output in {26}:
            return position == 28
        elif mlp_0_0_output in {27}:
            return position == 25
        elif mlp_0_0_output in {28}:
            return position == 11
        elif mlp_0_0_output in {29}:
            return position == 20
        elif mlp_0_0_output in {30}:
            return position == 14
        elif mlp_0_0_output in {32, 31}:
            return position == 6
        elif mlp_0_0_output in {35}:
            return position == 37
        elif mlp_0_0_output in {37}:
            return position == 31
        elif mlp_0_0_output in {38}:
            return position == 12

    attn_1_6_pattern = select_closest(positions, mlp_0_0_outputs, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_2_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(q_position, k_position):
        if q_position in {0, 5, 6}:
            return k_position == 3
        elif q_position in {1, 21}:
            return k_position == 4
        elif q_position in {25, 2, 33, 31}:
            return k_position == 1
        elif q_position in {3, 4}:
            return k_position == 2
        elif q_position in {14, 7}:
            return k_position == 16
        elif q_position in {8, 19}:
            return k_position == 17
        elif q_position in {9, 11}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {24, 12, 30}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 30
        elif q_position in {15}:
            return k_position == 12
        elif q_position in {16}:
            return k_position == 18
        elif q_position in {17}:
            return k_position == 8
        elif q_position in {18}:
            return k_position == 14
        elif q_position in {20, 23}:
            return k_position == 15
        elif q_position in {35, 36, 37, 22}:
            return k_position == 7
        elif q_position in {26}:
            return k_position == 23
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {28, 38, 39}:
            return k_position == 6
        elif q_position in {29}:
            return k_position == 28
        elif q_position in {32}:
            return k_position == 35
        elif q_position in {34}:
            return k_position == 29

    attn_1_7_pattern = select_closest(positions, positions, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, tokens)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, attn_0_6_output):
        if position in {0, 34, 14, 16, 17}:
            return attn_0_6_output == "</s>"
        elif position in {1, 2, 13}:
            return attn_0_6_output == "<s>"
        elif position in {3, 7, 8, 11, 12}:
            return attn_0_6_output == "1"
        elif position in {4}:
            return attn_0_6_output == "4"
        elif position in {5}:
            return attn_0_6_output == "2"
        elif position in {
            6,
            15,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            32,
            33,
            35,
            36,
            37,
        }:
            return attn_0_6_output == ""
        elif position in {38, 39, 9, 10, 30, 31}:
            return attn_0_6_output == "0"

    num_attn_1_0_pattern = select(attn_0_6_outputs, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_4_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(mlp_0_2_output, attn_0_5_output):
        if mlp_0_2_output in {0, 32, 33, 3, 4, 5, 6, 7, 34, 35, 37, 29, 30}:
            return attn_0_5_output == "1"
        elif mlp_0_2_output in {1}:
            return attn_0_5_output == "2"
        elif mlp_0_2_output in {9, 2, 19}:
            return attn_0_5_output == "<s>"
        elif mlp_0_2_output in {
            8,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            36,
            38,
        }:
            return attn_0_5_output == ""
        elif mlp_0_2_output in {39, 31}:
            return attn_0_5_output == "0"

    num_attn_1_1_pattern = select(attn_0_5_outputs, mlp_0_2_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_7_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, token):
        if position in {0, 32}:
            return token == "3"
        elif position in {1, 30}:
            return token == "2"
        elif position in {2, 34, 35, 37, 38, 7, 8, 9, 39, 29, 31}:
            return token == "0"
        elif position in {33, 3, 4, 36}:
            return token == "1"
        elif position in {5, 6, 11, 13, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28}:
            return token == ""
        elif position in {10, 15}:
            return token == "</s>"
        elif position in {16, 12, 14}:
            return token == "<s>"

    num_attn_1_2_pattern = select(tokens, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_4_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(mlp_0_2_output, token):
        if mlp_0_2_output in {0, 1, 3, 36, 37, 38, 39, 14, 15, 18, 30}:
            return token == "0"
        elif mlp_0_2_output in {33, 2}:
            return token == "<s>"
        elif mlp_0_2_output in {
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            16,
            17,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            29,
            31,
            32,
            34,
            35,
        }:
            return token == ""
        elif mlp_0_2_output in {19}:
            return token == "1"
        elif mlp_0_2_output in {28}:
            return token == "<pad>"

    num_attn_1_3_pattern = select(tokens, mlp_0_2_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_4_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(position, attn_0_5_output):
        if position in {0, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return attn_0_5_output == ""
        elif position in {1}:
            return attn_0_5_output == "<s>"
        elif position in {32, 2, 3, 4, 5, 6, 7, 8, 9, 10}:
            return attn_0_5_output == "0"
        elif position in {33, 34, 35, 37, 38, 39, 11, 30, 31}:
            return attn_0_5_output == "1"
        elif position in {12, 13, 14, 15}:
            return attn_0_5_output == "</s>"
        elif position in {36}:
            return attn_0_5_output == "2"

    num_attn_1_4_pattern = select(attn_0_5_outputs, positions, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_7_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(mlp_0_2_output, token):
        if mlp_0_2_output in {0, 4}:
            return token == "</s>"
        elif mlp_0_2_output in {1, 35, 36, 37, 14, 18, 24}:
            return token == "0"
        elif mlp_0_2_output in {32, 2, 3, 5, 7, 10, 13, 19, 20, 21, 22, 23, 25}:
            return token == ""
        elif mlp_0_2_output in {33, 34, 29, 6}:
            return token == "<s>"
        elif mlp_0_2_output in {38, 39, 8, 9, 11, 12, 15, 17, 26, 28, 30, 31}:
            return token == "1"
        elif mlp_0_2_output in {16}:
            return token == "2"
        elif mlp_0_2_output in {27}:
            return token == "3"

    num_attn_1_5_pattern = select(tokens, mlp_0_2_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_1_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(position, attn_0_7_output):
        if position in {
            0,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
        }:
            return attn_0_7_output == ""
        elif position in {1, 2, 3, 4, 5, 6, 7, 8, 34, 35, 36, 37, 38, 39, 29, 30, 31}:
            return attn_0_7_output == "0"
        elif position in {32, 9, 33}:
            return attn_0_7_output == "1"
        elif position in {10, 21}:
            return attn_0_7_output == "<pad>"

    num_attn_1_6_pattern = select(attn_0_7_outputs, positions, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_4_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(attn_0_0_output, attn_0_5_output):
        if attn_0_0_output in {"3", "<s>", "0"}:
            return attn_0_5_output == "1"
        elif attn_0_0_output in {"1"}:
            return attn_0_5_output == "</s>"
        elif attn_0_0_output in {"2", "4", "</s>"}:
            return attn_0_5_output == ""

    num_attn_1_7_pattern = select(attn_0_5_outputs, attn_0_0_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_4_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(token, attn_1_4_output):
        key = (token, attn_1_4_output)
        if key in {("<s>", "<s>")}:
            return 39
        return 33

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(tokens, attn_1_4_outputs)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_6_output, attn_1_7_output):
        key = (attn_1_6_output, attn_1_7_output)
        return 29

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_6_outputs, attn_1_7_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(position):
        key = position
        if key in {1, 2, 29}:
            return 34
        return 22

    mlp_1_2_outputs = [mlp_1_2(k0) for k0 in positions]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(position, attn_1_2_output):
        key = (position, attn_1_2_output)
        if key in {
            (0, "0"),
            (0, "2"),
            (0, "4"),
            (0, "</s>"),
            (0, "<s>"),
            (1, "0"),
            (1, "</s>"),
            (1, "<s>"),
            (2, "0"),
            (2, "4"),
            (2, "</s>"),
            (2, "<s>"),
            (3, "0"),
            (3, "1"),
            (3, "2"),
            (3, "4"),
            (3, "</s>"),
            (3, "<s>"),
            (4, "0"),
            (4, "4"),
            (4, "</s>"),
            (4, "<s>"),
            (7, "0"),
            (7, "1"),
            (7, "2"),
            (7, "4"),
            (7, "</s>"),
            (7, "<s>"),
            (21, "0"),
            (21, "1"),
            (21, "4"),
            (21, "</s>"),
            (22, "0"),
            (22, "</s>"),
            (29, "0"),
            (29, "2"),
            (29, "4"),
            (29, "</s>"),
            (29, "<s>"),
            (30, "0"),
            (32, "0"),
        }:
            return 5
        elif key in {
            (0, "3"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (2, "2"),
            (2, "3"),
            (3, "3"),
            (4, "2"),
            (4, "3"),
            (7, "3"),
            (29, "3"),
            (30, "3"),
            (31, "3"),
            (32, "3"),
            (33, "3"),
            (34, "3"),
            (35, "3"),
            (36, "3"),
            (37, "3"),
            (38, "3"),
            (39, "3"),
        }:
            return 35
        elif key in {(0, "1"), (1, "1"), (2, "1"), (4, "1"), (29, "1")}:
            return 13
        return 11

    mlp_1_3_outputs = [mlp_1_3(k0, k1) for k0, k1 in zip(positions, attn_1_2_outputs)]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_1_output, num_attn_1_5_output):
        key = (num_attn_1_1_output, num_attn_1_5_output)
        if key in {(0, 0)}:
            return 22
        return 32

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_1_output, num_attn_1_7_output):
        key = (num_attn_1_1_output, num_attn_1_7_output)
        return 12

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_7_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_4_output, num_attn_1_1_output):
        key = (num_attn_1_4_output, num_attn_1_1_output)
        return 39

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_7_output, num_attn_0_2_output):
        key = (num_attn_1_7_output, num_attn_0_2_output)
        return 39

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(token, attn_0_5_output):
        if token in {"2", "0"}:
            return attn_0_5_output == "4"
        elif token in {"4", "1", "</s>"}:
            return attn_0_5_output == ""
        elif token in {"3", "<s>"}:
            return attn_0_5_output == "<s>"

    attn_2_0_pattern = select_closest(attn_0_5_outputs, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_4_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_1_1_output, attn_0_7_output):
        if attn_1_1_output in {"0"}:
            return attn_0_7_output == "3"
        elif attn_1_1_output in {"1"}:
            return attn_0_7_output == "<s>"
        elif attn_1_1_output in {"2", "4", "</s>", "3"}:
            return attn_0_7_output == ""
        elif attn_1_1_output in {"<s>"}:
            return attn_0_7_output == "<pad>"

    attn_2_1_pattern = select_closest(attn_0_7_outputs, attn_1_1_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_1_3_output, mlp_1_3_output):
        if attn_1_3_output in {"0"}:
            return mlp_1_3_output == 39
        elif attn_1_3_output in {"<s>", "1"}:
            return mlp_1_3_output == 26
        elif attn_1_3_output in {"2"}:
            return mlp_1_3_output == 1
        elif attn_1_3_output in {"3"}:
            return mlp_1_3_output == 25
        elif attn_1_3_output in {"4"}:
            return mlp_1_3_output == 0
        elif attn_1_3_output in {"</s>"}:
            return mlp_1_3_output == 9

    attn_2_2_pattern = select_closest(mlp_1_3_outputs, attn_1_3_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "3"
        elif q_token in {"1"}:
            return k_token == "2"
        elif q_token in {"2", "4"}:
            return k_token == "0"
        elif q_token in {"3"}:
            return k_token == "4"
        elif q_token in {"<s>", "</s>"}:
            return k_token == ""

    attn_2_3_pattern = select_closest(tokens, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_7_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(q_token, k_token):
        if q_token in {"2", "1", "0"}:
            return k_token == "4"
        elif q_token in {"3"}:
            return k_token == "0"
        elif q_token in {"4"}:
            return k_token == "1"
        elif q_token in {"<s>", "</s>"}:
            return k_token == ""

    attn_2_4_pattern = select_closest(tokens, tokens, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_1_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(mlp_0_2_output, position):
        if mlp_0_2_output in {0, 9}:
            return position == 29
        elif mlp_0_2_output in {1}:
            return position == 16
        elif mlp_0_2_output in {2, 37, 22, 23}:
            return position == 6
        elif mlp_0_2_output in {3}:
            return position == 35
        elif mlp_0_2_output in {34, 27, 4}:
            return position == 1
        elif mlp_0_2_output in {33, 5, 10, 19, 31}:
            return position == 7
        elif mlp_0_2_output in {32, 6}:
            return position == 38
        elif mlp_0_2_output in {26, 13, 7}:
            return position == 5
        elif mlp_0_2_output in {8, 25}:
            return position == 32
        elif mlp_0_2_output in {35, 11}:
            return position == 25
        elif mlp_0_2_output in {12}:
            return position == 31
        elif mlp_0_2_output in {14}:
            return position == 28
        elif mlp_0_2_output in {36, 15}:
            return position == 19
        elif mlp_0_2_output in {16}:
            return position == 8
        elif mlp_0_2_output in {17}:
            return position == 3
        elif mlp_0_2_output in {18}:
            return position == 39
        elif mlp_0_2_output in {20, 29}:
            return position == 4
        elif mlp_0_2_output in {21}:
            return position == 26
        elif mlp_0_2_output in {24}:
            return position == 22
        elif mlp_0_2_output in {28}:
            return position == 24
        elif mlp_0_2_output in {30}:
            return position == 2
        elif mlp_0_2_output in {38, 39}:
            return position == 12

    attn_2_5_pattern = select_closest(positions, mlp_0_2_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_1_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(q_token, k_token):
        if q_token in {"2", "1", "0"}:
            return k_token == "4"
        elif q_token in {"3", "<s>", "</s>"}:
            return k_token == ""
        elif q_token in {"4"}:
            return k_token == "<s>"

    attn_2_6_pattern = select_closest(tokens, tokens, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_2_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(q_mlp_0_2_output, k_mlp_0_2_output):
        if q_mlp_0_2_output in {0, 8}:
            return k_mlp_0_2_output == 1
        elif q_mlp_0_2_output in {32, 1, 12, 13, 15, 16}:
            return k_mlp_0_2_output == 30
        elif q_mlp_0_2_output in {2, 3, 37, 9, 22, 23}:
            return k_mlp_0_2_output == 34
        elif q_mlp_0_2_output in {10, 4, 31}:
            return k_mlp_0_2_output == 29
        elif q_mlp_0_2_output in {24, 5}:
            return k_mlp_0_2_output == 6
        elif q_mlp_0_2_output in {6}:
            return k_mlp_0_2_output == 17
        elif q_mlp_0_2_output in {7}:
            return k_mlp_0_2_output == 25
        elif q_mlp_0_2_output in {11}:
            return k_mlp_0_2_output == 10
        elif q_mlp_0_2_output in {14}:
            return k_mlp_0_2_output == 14
        elif q_mlp_0_2_output in {17}:
            return k_mlp_0_2_output == 22
        elif q_mlp_0_2_output in {18}:
            return k_mlp_0_2_output == 31
        elif q_mlp_0_2_output in {39, 19, 25, 28, 30}:
            return k_mlp_0_2_output == 13
        elif q_mlp_0_2_output in {20}:
            return k_mlp_0_2_output == 5
        elif q_mlp_0_2_output in {21}:
            return k_mlp_0_2_output == 26
        elif q_mlp_0_2_output in {26}:
            return k_mlp_0_2_output == 12
        elif q_mlp_0_2_output in {27}:
            return k_mlp_0_2_output == 7
        elif q_mlp_0_2_output in {29}:
            return k_mlp_0_2_output == 33
        elif q_mlp_0_2_output in {33}:
            return k_mlp_0_2_output == 11
        elif q_mlp_0_2_output in {34}:
            return k_mlp_0_2_output == 36
        elif q_mlp_0_2_output in {35}:
            return k_mlp_0_2_output == 24
        elif q_mlp_0_2_output in {36}:
            return k_mlp_0_2_output == 35
        elif q_mlp_0_2_output in {38}:
            return k_mlp_0_2_output == 19

    attn_2_7_pattern = select_closest(mlp_0_2_outputs, mlp_0_2_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_7_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_0_2_output, token):
        if mlp_0_2_output in {0, 2, 3, 39, 20}:
            return token == "</s>"
        elif mlp_0_2_output in {
            1,
            4,
            5,
            6,
            7,
            8,
            12,
            16,
            17,
            19,
            21,
            22,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            32,
            34,
            36,
            38,
        }:
            return token == "0"
        elif mlp_0_2_output in {35, 9, 10, 11, 13, 18, 23}:
            return token == ""
        elif mlp_0_2_output in {33, 37, 14, 15}:
            return token == "<s>"
        elif mlp_0_2_output in {31}:
            return token == "<pad>"

    num_attn_2_0_pattern = select(tokens, mlp_0_2_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_1_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_2_output, token):
        if attn_1_2_output in {"2", "<s>", "</s>", "0"}:
            return token == "2"
        elif attn_1_2_output in {"3", "1", "4"}:
            return token == ""

    num_attn_2_1_pattern = select(tokens, attn_1_2_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_4_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(mlp_0_2_output, attn_1_1_output):
        if mlp_0_2_output in {
            0,
            4,
            5,
            6,
            7,
            12,
            13,
            14,
            16,
            18,
            21,
            26,
            27,
            28,
            29,
            32,
            34,
            35,
            36,
            38,
        }:
            return attn_1_1_output == ""
        elif mlp_0_2_output in {1, 2, 3, 9, 10, 11, 19, 20, 22, 23, 30, 31}:
            return attn_1_1_output == "0"
        elif mlp_0_2_output in {37, 39, 8, 15, 17, 25}:
            return attn_1_1_output == "</s>"
        elif mlp_0_2_output in {24}:
            return attn_1_1_output == "3"
        elif mlp_0_2_output in {33}:
            return attn_1_1_output == "1"

    num_attn_2_2_pattern = select(attn_1_1_outputs, mlp_0_2_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_0_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(position, attn_0_7_output):
        if position in {0, 32, 2, 33, 4, 5, 35, 7, 8, 9, 11, 12, 13}:
            return attn_0_7_output == "1"
        elif position in {1, 3, 30}:
            return attn_0_7_output == "2"
        elif position in {6}:
            return attn_0_7_output == "3"
        elif position in {34, 36, 37, 38, 39, 10, 31}:
            return attn_0_7_output == "0"
        elif position in {17, 14, 15}:
            return attn_0_7_output == "</s>"
        elif position in {16, 18, 29}:
            return attn_0_7_output == "<s>"
        elif position in {19, 20, 21, 22, 23, 24, 25, 26, 27, 28}:
            return attn_0_7_output == ""

    num_attn_2_3_pattern = select(attn_0_7_outputs, positions, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_7_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_1_0_output, attn_1_1_output):
        if attn_1_0_output in {"4", "1", "</s>", "0"}:
            return attn_1_1_output == ""
        elif attn_1_0_output in {"2"}:
            return attn_1_1_output == "</s>"
        elif attn_1_0_output in {"3"}:
            return attn_1_1_output == "2"
        elif attn_1_0_output in {"<s>"}:
            return attn_1_1_output == "<s>"

    num_attn_2_4_pattern = select(attn_1_1_outputs, attn_1_0_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_4_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(position, attn_1_7_output):
        if position in {0}:
            return attn_1_7_output == "0"
        elif position in {1, 14, 15, 16, 17, 18, 19}:
            return attn_1_7_output == "</s>"
        elif position in {32, 33, 2, 35, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 37, 30, 31}:
            return attn_1_7_output == "2"
        elif position in {3}:
            return attn_1_7_output == "3"
        elif position in {34, 36, 38, 39, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return attn_1_7_output == ""

    num_attn_2_5_pattern = select(attn_1_7_outputs, positions, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_7_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_0_output, attn_0_1_output):
        if attn_1_0_output in {"3", "0"}:
            return attn_0_1_output == "<s>"
        elif attn_1_0_output in {"1"}:
            return attn_0_1_output == "0"
        elif attn_1_0_output in {"2", "<s>", "4"}:
            return attn_0_1_output == "1"
        elif attn_1_0_output in {"</s>"}:
            return attn_0_1_output == "2"

    num_attn_2_6_pattern = select(attn_0_1_outputs, attn_1_0_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_7_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_1_0_output, token):
        if attn_1_0_output in {"2", "<s>", "</s>", "0"}:
            return token == "0"
        elif attn_1_0_output in {"1"}:
            return token == "</s>"
        elif attn_1_0_output in {"3"}:
            return token == "<s>"
        elif attn_1_0_output in {"4"}:
            return token == "2"

    num_attn_2_7_pattern = select(tokens, attn_1_0_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_7_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_2_output, position):
        key = (attn_2_2_output, position)
        if key in {
            ("0", 4),
            ("0", 5),
            ("0", 6),
            ("0", 7),
            ("0", 21),
            ("0", 22),
            ("0", 23),
            ("0", 25),
            ("0", 26),
            ("0", 27),
            ("0", 28),
            ("1", 4),
            ("1", 5),
            ("1", 6),
            ("1", 7),
            ("1", 21),
            ("1", 22),
            ("1", 23),
            ("1", 25),
            ("1", 26),
            ("1", 27),
            ("1", 28),
            ("2", 4),
            ("2", 5),
            ("2", 6),
            ("2", 7),
            ("2", 21),
            ("2", 22),
            ("2", 23),
            ("2", 25),
            ("2", 26),
            ("2", 27),
            ("2", 28),
            ("3", 4),
            ("3", 5),
            ("3", 6),
            ("3", 7),
            ("3", 21),
            ("3", 22),
            ("3", 23),
            ("3", 25),
            ("3", 26),
            ("3", 27),
            ("3", 28),
            ("4", 4),
            ("4", 5),
            ("4", 6),
            ("4", 7),
            ("4", 21),
            ("4", 22),
            ("4", 23),
            ("4", 25),
            ("4", 26),
            ("4", 27),
            ("4", 28),
            ("</s>", 4),
            ("</s>", 5),
            ("</s>", 6),
            ("</s>", 7),
            ("</s>", 21),
            ("</s>", 22),
            ("</s>", 23),
            ("</s>", 25),
            ("</s>", 26),
            ("</s>", 27),
            ("</s>", 28),
            ("<s>", 4),
            ("<s>", 5),
            ("<s>", 6),
            ("<s>", 7),
            ("<s>", 21),
            ("<s>", 22),
            ("<s>", 23),
            ("<s>", 25),
            ("<s>", 26),
            ("<s>", 27),
            ("<s>", 28),
        }:
            return 5
        elif key in {
            ("0", 0),
            ("0", 1),
            ("0", 2),
            ("0", 3),
            ("0", 8),
            ("0", 14),
            ("0", 15),
            ("0", 16),
            ("0", 29),
            ("0", 30),
            ("0", 31),
            ("0", 32),
            ("0", 33),
            ("0", 34),
            ("0", 35),
            ("0", 36),
            ("0", 37),
            ("0", 38),
            ("0", 39),
            ("1", 0),
            ("1", 1),
            ("1", 2),
            ("1", 3),
            ("1", 8),
            ("1", 14),
            ("1", 15),
            ("1", 16),
            ("1", 29),
            ("1", 30),
            ("1", 31),
            ("1", 32),
            ("1", 33),
            ("1", 34),
            ("1", 35),
            ("1", 36),
            ("1", 37),
            ("1", 38),
            ("1", 39),
            ("2", 1),
            ("2", 2),
            ("2", 3),
            ("2", 8),
            ("3", 1),
            ("3", 2),
            ("3", 3),
            ("3", 8),
            ("4", 1),
            ("4", 2),
            ("4", 3),
            ("4", 8),
            ("</s>", 1),
            ("</s>", 2),
            ("</s>", 3),
            ("</s>", 8),
            ("<s>", 1),
            ("<s>", 2),
            ("<s>", 3),
            ("<s>", 8),
        }:
            return 10
        return 9

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_2_outputs, positions)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_3_output, position):
        key = (attn_2_3_output, position)
        if key in {
            ("0", 8),
            ("1", 0),
            ("1", 1),
            ("1", 2),
            ("1", 4),
            ("1", 8),
            ("1", 11),
            ("1", 12),
            ("1", 14),
            ("1", 16),
            ("1", 17),
            ("1", 18),
            ("1", 19),
            ("1", 20),
            ("1", 29),
            ("1", 30),
            ("1", 31),
            ("1", 32),
            ("1", 33),
            ("1", 34),
            ("1", 35),
            ("1", 36),
            ("1", 37),
            ("1", 38),
            ("1", 39),
            ("2", 8),
            ("3", 8),
            ("4", 8),
            ("</s>", 8),
            ("<s>", 8),
            ("<s>", 14),
            ("<s>", 20),
        }:
            return 1
        elif key in {
            ("0", 20),
            ("0", 22),
            ("1", 22),
            ("1", 26),
            ("2", 22),
            ("3", 22),
            ("4", 22),
            ("</s>", 22),
            ("<s>", 22),
        }:
            return 9
        return 23

    mlp_2_1_outputs = [mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_3_outputs, positions)]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(attn_2_4_output, attn_2_6_output):
        key = (attn_2_4_output, attn_2_6_output)
        return 35

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(attn_2_4_outputs, attn_2_6_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(num_mlp_1_0_output, num_mlp_1_1_output):
        key = (num_mlp_1_0_output, num_mlp_1_1_output)
        return 23

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(num_mlp_1_0_outputs, num_mlp_1_1_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_0_3_output):
        key = num_attn_0_3_output
        if key in {0}:
            return 28
        return 22

    num_mlp_2_0_outputs = [num_mlp_2_0(k0) for k0 in num_attn_0_3_outputs]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_7_output):
        key = num_attn_1_7_output
        return 7

    num_mlp_2_1_outputs = [num_mlp_2_1(k0) for k0 in num_attn_1_7_outputs]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_0_2_output, num_attn_0_4_output):
        key = (num_attn_0_2_output, num_attn_0_4_output)
        if key in {
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (4, 1),
            (5, 0),
            (5, 1),
            (6, 0),
            (6, 1),
            (7, 0),
            (7, 1),
            (8, 0),
            (8, 1),
            (9, 0),
            (9, 1),
            (10, 0),
            (10, 1),
            (11, 0),
            (11, 1),
            (12, 0),
            (12, 1),
            (12, 2),
            (13, 0),
            (13, 1),
            (13, 2),
            (14, 0),
            (14, 1),
            (14, 2),
            (15, 0),
            (15, 1),
            (15, 2),
            (16, 0),
            (16, 1),
            (16, 2),
            (17, 0),
            (17, 1),
            (17, 2),
            (18, 0),
            (18, 1),
            (18, 2),
            (19, 0),
            (19, 1),
            (19, 2),
            (20, 0),
            (20, 1),
            (20, 2),
            (21, 0),
            (21, 1),
            (21, 2),
            (21, 3),
            (22, 0),
            (22, 1),
            (22, 2),
            (22, 3),
            (23, 0),
            (23, 1),
            (23, 2),
            (23, 3),
            (24, 0),
            (24, 1),
            (24, 2),
            (24, 3),
            (25, 0),
            (25, 1),
            (25, 2),
            (25, 3),
            (26, 0),
            (26, 1),
            (26, 2),
            (26, 3),
            (27, 0),
            (27, 1),
            (27, 2),
            (27, 3),
            (28, 0),
            (28, 1),
            (28, 2),
            (28, 3),
            (29, 0),
            (29, 1),
            (29, 2),
            (29, 3),
            (30, 0),
            (30, 1),
            (30, 2),
            (30, 3),
            (30, 4),
            (31, 0),
            (31, 1),
            (31, 2),
            (31, 3),
            (31, 4),
            (32, 0),
            (32, 1),
            (32, 2),
            (32, 3),
            (32, 4),
            (33, 0),
            (33, 1),
            (33, 2),
            (33, 3),
            (33, 4),
            (34, 0),
            (34, 1),
            (34, 2),
            (34, 3),
            (34, 4),
            (35, 0),
            (35, 1),
            (35, 2),
            (35, 3),
            (35, 4),
            (36, 0),
            (36, 1),
            (36, 2),
            (36, 3),
            (36, 4),
            (37, 0),
            (37, 1),
            (37, 2),
            (37, 3),
            (37, 4),
            (38, 0),
            (38, 1),
            (38, 2),
            (38, 3),
            (38, 4),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
            (39, 4),
            (39, 5),
            (40, 0),
            (40, 1),
            (40, 2),
            (40, 3),
            (40, 4),
            (40, 5),
            (41, 0),
            (41, 1),
            (41, 2),
            (41, 3),
            (41, 4),
            (41, 5),
            (42, 0),
            (42, 1),
            (42, 2),
            (42, 3),
            (42, 4),
            (42, 5),
            (43, 0),
            (43, 1),
            (43, 2),
            (43, 3),
            (43, 4),
            (43, 5),
            (44, 0),
            (44, 1),
            (44, 2),
            (44, 3),
            (44, 4),
            (44, 5),
            (45, 0),
            (45, 1),
            (45, 2),
            (45, 3),
            (45, 4),
            (45, 5),
            (46, 0),
            (46, 1),
            (46, 2),
            (46, 3),
            (46, 4),
            (46, 5),
            (47, 0),
            (47, 1),
            (47, 2),
            (47, 3),
            (47, 4),
            (47, 5),
            (48, 0),
            (48, 1),
            (48, 2),
            (48, 3),
            (48, 4),
            (48, 5),
            (48, 6),
            (49, 0),
            (49, 1),
            (49, 2),
            (49, 3),
            (49, 4),
            (49, 5),
            (49, 6),
            (50, 0),
            (50, 1),
            (50, 2),
            (50, 3),
            (50, 4),
            (50, 5),
            (50, 6),
            (51, 0),
            (51, 1),
            (51, 2),
            (51, 3),
            (51, 4),
            (51, 5),
            (51, 6),
            (52, 0),
            (52, 1),
            (52, 2),
            (52, 3),
            (52, 4),
            (52, 5),
            (52, 6),
            (53, 0),
            (53, 1),
            (53, 2),
            (53, 3),
            (53, 4),
            (53, 5),
            (53, 6),
            (54, 0),
            (54, 1),
            (54, 2),
            (54, 3),
            (54, 4),
            (54, 5),
            (54, 6),
            (55, 0),
            (55, 1),
            (55, 2),
            (55, 3),
            (55, 4),
            (55, 5),
            (55, 6),
            (56, 0),
            (56, 1),
            (56, 2),
            (56, 3),
            (56, 4),
            (56, 5),
            (56, 6),
            (57, 0),
            (57, 1),
            (57, 2),
            (57, 3),
            (57, 4),
            (57, 5),
            (57, 6),
            (57, 7),
            (58, 0),
            (58, 1),
            (58, 2),
            (58, 3),
            (58, 4),
            (58, 5),
            (58, 6),
            (58, 7),
            (59, 0),
            (59, 1),
            (59, 2),
            (59, 3),
            (59, 4),
            (59, 5),
            (59, 6),
            (59, 7),
            (60, 0),
            (60, 1),
            (60, 2),
            (60, 3),
            (60, 4),
            (60, 5),
            (60, 6),
            (60, 7),
            (61, 0),
            (61, 1),
            (61, 2),
            (61, 3),
            (61, 4),
            (61, 5),
            (61, 6),
            (61, 7),
            (62, 0),
            (62, 1),
            (62, 2),
            (62, 3),
            (62, 4),
            (62, 5),
            (62, 6),
            (62, 7),
            (63, 0),
            (63, 1),
            (63, 2),
            (63, 3),
            (63, 4),
            (63, 5),
            (63, 6),
            (63, 7),
            (64, 0),
            (64, 1),
            (64, 2),
            (64, 3),
            (64, 4),
            (64, 5),
            (64, 6),
            (64, 7),
            (65, 0),
            (65, 1),
            (65, 2),
            (65, 3),
            (65, 4),
            (65, 5),
            (65, 6),
            (65, 7),
            (66, 0),
            (66, 1),
            (66, 2),
            (66, 3),
            (66, 4),
            (66, 5),
            (66, 6),
            (66, 7),
            (66, 8),
            (67, 0),
            (67, 1),
            (67, 2),
            (67, 3),
            (67, 4),
            (67, 5),
            (67, 6),
            (67, 7),
            (67, 8),
            (68, 0),
            (68, 1),
            (68, 2),
            (68, 3),
            (68, 4),
            (68, 5),
            (68, 6),
            (68, 7),
            (68, 8),
            (69, 0),
            (69, 1),
            (69, 2),
            (69, 3),
            (69, 4),
            (69, 5),
            (69, 6),
            (69, 7),
            (69, 8),
            (70, 0),
            (70, 1),
            (70, 2),
            (70, 3),
            (70, 4),
            (70, 5),
            (70, 6),
            (70, 7),
            (70, 8),
            (71, 0),
            (71, 1),
            (71, 2),
            (71, 3),
            (71, 4),
            (71, 5),
            (71, 6),
            (71, 7),
            (71, 8),
            (72, 0),
            (72, 1),
            (72, 2),
            (72, 3),
            (72, 4),
            (72, 5),
            (72, 6),
            (72, 7),
            (72, 8),
            (73, 0),
            (73, 1),
            (73, 2),
            (73, 3),
            (73, 4),
            (73, 5),
            (73, 6),
            (73, 7),
            (73, 8),
            (74, 0),
            (74, 1),
            (74, 2),
            (74, 3),
            (74, 4),
            (74, 5),
            (74, 6),
            (74, 7),
            (74, 8),
            (75, 0),
            (75, 1),
            (75, 2),
            (75, 3),
            (75, 4),
            (75, 5),
            (75, 6),
            (75, 7),
            (75, 8),
            (75, 9),
            (76, 0),
            (76, 1),
            (76, 2),
            (76, 3),
            (76, 4),
            (76, 5),
            (76, 6),
            (76, 7),
            (76, 8),
            (76, 9),
            (77, 0),
            (77, 1),
            (77, 2),
            (77, 3),
            (77, 4),
            (77, 5),
            (77, 6),
            (77, 7),
            (77, 8),
            (77, 9),
            (78, 0),
            (78, 1),
            (78, 2),
            (78, 3),
            (78, 4),
            (78, 5),
            (78, 6),
            (78, 7),
            (78, 8),
            (78, 9),
            (79, 0),
            (79, 1),
            (79, 2),
            (79, 3),
            (79, 4),
            (79, 5),
            (79, 6),
            (79, 7),
            (79, 8),
            (79, 9),
            (80, 0),
            (80, 1),
            (80, 2),
            (80, 3),
            (80, 4),
            (80, 5),
            (80, 6),
            (80, 7),
            (80, 8),
            (80, 9),
            (81, 0),
            (81, 1),
            (81, 2),
            (81, 3),
            (81, 4),
            (81, 5),
            (81, 6),
            (81, 7),
            (81, 8),
            (81, 9),
            (82, 0),
            (82, 1),
            (82, 2),
            (82, 3),
            (82, 4),
            (82, 5),
            (82, 6),
            (82, 7),
            (82, 8),
            (82, 9),
            (83, 0),
            (83, 1),
            (83, 2),
            (83, 3),
            (83, 4),
            (83, 5),
            (83, 6),
            (83, 7),
            (83, 8),
            (83, 9),
            (84, 0),
            (84, 1),
            (84, 2),
            (84, 3),
            (84, 4),
            (84, 5),
            (84, 6),
            (84, 7),
            (84, 8),
            (84, 9),
            (84, 10),
            (85, 0),
            (85, 1),
            (85, 2),
            (85, 3),
            (85, 4),
            (85, 5),
            (85, 6),
            (85, 7),
            (85, 8),
            (85, 9),
            (85, 10),
            (86, 0),
            (86, 1),
            (86, 2),
            (86, 3),
            (86, 4),
            (86, 5),
            (86, 6),
            (86, 7),
            (86, 8),
            (86, 9),
            (86, 10),
            (87, 0),
            (87, 1),
            (87, 2),
            (87, 3),
            (87, 4),
            (87, 5),
            (87, 6),
            (87, 7),
            (87, 8),
            (87, 9),
            (87, 10),
            (88, 0),
            (88, 1),
            (88, 2),
            (88, 3),
            (88, 4),
            (88, 5),
            (88, 6),
            (88, 7),
            (88, 8),
            (88, 9),
            (88, 10),
            (89, 0),
            (89, 1),
            (89, 2),
            (89, 3),
            (89, 4),
            (89, 5),
            (89, 6),
            (89, 7),
            (89, 8),
            (89, 9),
            (89, 10),
            (90, 0),
            (90, 1),
            (90, 2),
            (90, 3),
            (90, 4),
            (90, 5),
            (90, 6),
            (90, 7),
            (90, 8),
            (90, 9),
            (90, 10),
            (91, 0),
            (91, 1),
            (91, 2),
            (91, 3),
            (91, 4),
            (91, 5),
            (91, 6),
            (91, 7),
            (91, 8),
            (91, 9),
            (91, 10),
            (92, 0),
            (92, 1),
            (92, 2),
            (92, 3),
            (92, 4),
            (92, 5),
            (92, 6),
            (92, 7),
            (92, 8),
            (92, 9),
            (92, 10),
            (93, 0),
            (93, 1),
            (93, 2),
            (93, 3),
            (93, 4),
            (93, 5),
            (93, 6),
            (93, 7),
            (93, 8),
            (93, 9),
            (93, 10),
            (93, 11),
            (94, 0),
            (94, 1),
            (94, 2),
            (94, 3),
            (94, 4),
            (94, 5),
            (94, 6),
            (94, 7),
            (94, 8),
            (94, 9),
            (94, 10),
            (94, 11),
            (95, 0),
            (95, 1),
            (95, 2),
            (95, 3),
            (95, 4),
            (95, 5),
            (95, 6),
            (95, 7),
            (95, 8),
            (95, 9),
            (95, 10),
            (95, 11),
            (96, 0),
            (96, 1),
            (96, 2),
            (96, 3),
            (96, 4),
            (96, 5),
            (96, 6),
            (96, 7),
            (96, 8),
            (96, 9),
            (96, 10),
            (96, 11),
            (97, 0),
            (97, 1),
            (97, 2),
            (97, 3),
            (97, 4),
            (97, 5),
            (97, 6),
            (97, 7),
            (97, 8),
            (97, 9),
            (97, 10),
            (97, 11),
            (98, 0),
            (98, 1),
            (98, 2),
            (98, 3),
            (98, 4),
            (98, 5),
            (98, 6),
            (98, 7),
            (98, 8),
            (98, 9),
            (98, 10),
            (98, 11),
            (99, 0),
            (99, 1),
            (99, 2),
            (99, 3),
            (99, 4),
            (99, 5),
            (99, 6),
            (99, 7),
            (99, 8),
            (99, 9),
            (99, 10),
            (99, 11),
            (100, 0),
            (100, 1),
            (100, 2),
            (100, 3),
            (100, 4),
            (100, 5),
            (100, 6),
            (100, 7),
            (100, 8),
            (100, 9),
            (100, 10),
            (100, 11),
            (101, 0),
            (101, 1),
            (101, 2),
            (101, 3),
            (101, 4),
            (101, 5),
            (101, 6),
            (101, 7),
            (101, 8),
            (101, 9),
            (101, 10),
            (101, 11),
            (102, 0),
            (102, 1),
            (102, 2),
            (102, 3),
            (102, 4),
            (102, 5),
            (102, 6),
            (102, 7),
            (102, 8),
            (102, 9),
            (102, 10),
            (102, 11),
            (102, 12),
            (103, 0),
            (103, 1),
            (103, 2),
            (103, 3),
            (103, 4),
            (103, 5),
            (103, 6),
            (103, 7),
            (103, 8),
            (103, 9),
            (103, 10),
            (103, 11),
            (103, 12),
            (104, 0),
            (104, 1),
            (104, 2),
            (104, 3),
            (104, 4),
            (104, 5),
            (104, 6),
            (104, 7),
            (104, 8),
            (104, 9),
            (104, 10),
            (104, 11),
            (104, 12),
            (105, 0),
            (105, 1),
            (105, 2),
            (105, 3),
            (105, 4),
            (105, 5),
            (105, 6),
            (105, 7),
            (105, 8),
            (105, 9),
            (105, 10),
            (105, 11),
            (105, 12),
            (106, 0),
            (106, 1),
            (106, 2),
            (106, 3),
            (106, 4),
            (106, 5),
            (106, 6),
            (106, 7),
            (106, 8),
            (106, 9),
            (106, 10),
            (106, 11),
            (106, 12),
            (107, 0),
            (107, 1),
            (107, 2),
            (107, 3),
            (107, 4),
            (107, 5),
            (107, 6),
            (107, 7),
            (107, 8),
            (107, 9),
            (107, 10),
            (107, 11),
            (107, 12),
            (108, 0),
            (108, 1),
            (108, 2),
            (108, 3),
            (108, 4),
            (108, 5),
            (108, 6),
            (108, 7),
            (108, 8),
            (108, 9),
            (108, 10),
            (108, 11),
            (108, 12),
            (109, 0),
            (109, 1),
            (109, 2),
            (109, 3),
            (109, 4),
            (109, 5),
            (109, 6),
            (109, 7),
            (109, 8),
            (109, 9),
            (109, 10),
            (109, 11),
            (109, 12),
            (110, 0),
            (110, 1),
            (110, 2),
            (110, 3),
            (110, 4),
            (110, 5),
            (110, 6),
            (110, 7),
            (110, 8),
            (110, 9),
            (110, 10),
            (110, 11),
            (110, 12),
            (111, 0),
            (111, 1),
            (111, 2),
            (111, 3),
            (111, 4),
            (111, 5),
            (111, 6),
            (111, 7),
            (111, 8),
            (111, 9),
            (111, 10),
            (111, 11),
            (111, 12),
            (111, 13),
            (112, 0),
            (112, 1),
            (112, 2),
            (112, 3),
            (112, 4),
            (112, 5),
            (112, 6),
            (112, 7),
            (112, 8),
            (112, 9),
            (112, 10),
            (112, 11),
            (112, 12),
            (112, 13),
            (113, 0),
            (113, 1),
            (113, 2),
            (113, 3),
            (113, 4),
            (113, 5),
            (113, 6),
            (113, 7),
            (113, 8),
            (113, 9),
            (113, 10),
            (113, 11),
            (113, 12),
            (113, 13),
            (114, 0),
            (114, 1),
            (114, 2),
            (114, 3),
            (114, 4),
            (114, 5),
            (114, 6),
            (114, 7),
            (114, 8),
            (114, 9),
            (114, 10),
            (114, 11),
            (114, 12),
            (114, 13),
            (115, 0),
            (115, 1),
            (115, 2),
            (115, 3),
            (115, 4),
            (115, 5),
            (115, 6),
            (115, 7),
            (115, 8),
            (115, 9),
            (115, 10),
            (115, 11),
            (115, 12),
            (115, 13),
            (116, 0),
            (116, 1),
            (116, 2),
            (116, 3),
            (116, 4),
            (116, 5),
            (116, 6),
            (116, 7),
            (116, 8),
            (116, 9),
            (116, 10),
            (116, 11),
            (116, 12),
            (116, 13),
            (117, 0),
            (117, 1),
            (117, 2),
            (117, 3),
            (117, 4),
            (117, 5),
            (117, 6),
            (117, 7),
            (117, 8),
            (117, 9),
            (117, 10),
            (117, 11),
            (117, 12),
            (117, 13),
            (118, 0),
            (118, 1),
            (118, 2),
            (118, 3),
            (118, 4),
            (118, 5),
            (118, 6),
            (118, 7),
            (118, 8),
            (118, 9),
            (118, 10),
            (118, 11),
            (118, 12),
            (118, 13),
            (119, 0),
            (119, 1),
            (119, 2),
            (119, 3),
            (119, 4),
            (119, 5),
            (119, 6),
            (119, 7),
            (119, 8),
            (119, 9),
            (119, 10),
            (119, 11),
            (119, 12),
            (119, 13),
        }:
            return 4
        elif key in {
            (3, 1),
            (38, 5),
            (47, 6),
            (56, 7),
            (65, 8),
            (74, 9),
            (83, 10),
            (92, 11),
            (101, 12),
            (110, 13),
            (119, 14),
        }:
            return 33
        return 2

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_4_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_1_3_output):
        key = num_attn_1_3_output
        return 26

    num_mlp_2_3_outputs = [num_mlp_2_3(k0) for k0 in num_attn_1_3_outputs]
    num_mlp_2_3_output_scores = classifier_weights.loc[
        [("num_mlp_2_3_outputs", str(v)) for v in num_mlp_2_3_outputs]
    ]

    feature_logits = pd.concat(
        [
            df.reset_index()
            for df in [
                token_scores,
                position_scores,
                attn_0_0_output_scores,
                attn_0_1_output_scores,
                attn_0_2_output_scores,
                attn_0_3_output_scores,
                attn_0_4_output_scores,
                attn_0_5_output_scores,
                attn_0_6_output_scores,
                attn_0_7_output_scores,
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                mlp_0_2_output_scores,
                mlp_0_3_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                num_mlp_0_2_output_scores,
                num_mlp_0_3_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                attn_1_4_output_scores,
                attn_1_5_output_scores,
                attn_1_6_output_scores,
                attn_1_7_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                mlp_1_2_output_scores,
                mlp_1_3_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                num_mlp_1_2_output_scores,
                num_mlp_1_3_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                attn_2_4_output_scores,
                attn_2_5_output_scores,
                attn_2_6_output_scores,
                attn_2_7_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                mlp_2_2_output_scores,
                mlp_2_3_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                num_mlp_2_2_output_scores,
                num_mlp_2_3_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_0_4_output_scores,
                num_attn_0_5_output_scores,
                num_attn_0_6_output_scores,
                num_attn_0_7_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_1_4_output_scores,
                num_attn_1_5_output_scores,
                num_attn_1_6_output_scores,
                num_attn_1_7_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
                num_attn_2_4_output_scores,
                num_attn_2_5_output_scores,
                num_attn_2_6_output_scores,
                num_attn_2_7_output_scores,
            ]
        ]
    )
    logits = feature_logits.groupby(level=0).sum(numeric_only=True).to_numpy()
    classes = classifier_weights.columns.to_numpy()
    predictions = classes[logits.argmax(-1)]
    if tokens[0] == "<s>":
        predictions[0] = "<s>"
    if tokens[-1] == "</s>":
        predictions[-1] = "</s>"
    return predictions.tolist()


print(
    run(
        [
            "<s>",
            "1",
            "0",
            "0",
            "2",
            "1",
            "2",
            "4",
            "1",
            "0",
            "4",
            "2",
            "4",
            "2",
            "4",
            "3",
            "0",
            "1",
            "0",
            "2",
            "0",
            "1",
            "2",
            "2",
            "0",
            "0",
            "3",
            "2",
            "</s>",
        ]
    )
)
