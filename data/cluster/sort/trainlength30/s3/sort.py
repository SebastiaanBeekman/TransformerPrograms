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
        "output/length/rasp/sort/trainlength30/s3/sort_weights.csv",
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
    def predicate_0_0(position, token):
        if position in {0, 10, 15, 17, 20, 27}:
            return token == "0"
        elif position in {1}:
            return token == "2"
        elif position in {2, 11, 14, 16, 29}:
            return token == "1"
        elif position in {3, 4, 5, 6, 7, 8, 9, 13}:
            return token == "3"
        elif position in {12, 18, 19, 21, 23, 24, 25}:
            return token == "4"
        elif position in {28, 22}:
            return token == "</s>"
        elif position in {32, 33, 36, 37, 39, 26, 30, 31}:
            return token == ""
        elif position in {34, 38}:
            return token == "<pad>"
        elif position in {35}:
            return token == "<s>"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(position, token):
        if position in {0, 1, 4, 7, 13}:
            return token == "2"
        elif position in {2, 36, 39, 9, 10, 11, 16, 28}:
            return token == "0"
        elif position in {3, 12, 23, 15}:
            return token == "3"
        elif position in {5, 18, 19, 21, 22, 27}:
            return token == "4"
        elif position in {32, 33, 34, 35, 6, 38, 24, 30, 31}:
            return token == ""
        elif position in {8, 14, 17, 25, 26, 29}:
            return token == "1"
        elif position in {20, 37}:
            return token == "</s>"

    attn_0_1_pattern = select_closest(tokens, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 19}:
            return k_position == 20
        elif q_position in {1, 10, 33}:
            return k_position == 1
        elif q_position in {32, 2, 4}:
            return k_position == 5
        elif q_position in {9, 3}:
            return k_position == 10
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 15
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 16
        elif q_position in {17, 14}:
            return k_position == 18
        elif q_position in {15}:
            return k_position == 19
        elif q_position in {16}:
            return k_position == 17
        elif q_position in {18, 20}:
            return k_position == 23
        elif q_position in {21, 23}:
            return k_position == 26
        elif q_position in {22, 39}:
            return k_position == 27
        elif q_position in {24, 25, 26, 27}:
            return k_position == 28
        elif q_position in {28}:
            return k_position == 29
        elif q_position in {29}:
            return k_position == 2
        elif q_position in {30}:
            return k_position == 25
        elif q_position in {31}:
            return k_position == 37
        elif q_position in {34}:
            return k_position == 0
        elif q_position in {35}:
            return k_position == 11
        elif q_position in {36}:
            return k_position == 35
        elif q_position in {37, 38}:
            return k_position == 24

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 10
        elif q_position in {1, 3}:
            return k_position == 5
        elif q_position in {2, 7}:
            return k_position == 4
        elif q_position in {4, 6}:
            return k_position == 3
        elif q_position in {16, 25, 5}:
            return k_position == 9
        elif q_position in {8, 11}:
            return k_position == 1
        elif q_position in {9, 17}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 17
        elif q_position in {12}:
            return k_position == 26
        elif q_position in {28, 13}:
            return k_position == 14
        elif q_position in {14}:
            return k_position == 24
        elif q_position in {15}:
            return k_position == 22
        elif q_position in {18, 35}:
            return k_position == 21
        elif q_position in {19}:
            return k_position == 16
        elif q_position in {20}:
            return k_position == 12
        elif q_position in {21, 31}:
            return k_position == 0
        elif q_position in {24, 22}:
            return k_position == 28
        elif q_position in {23}:
            return k_position == 13
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {27}:
            return k_position == 20
        elif q_position in {29}:
            return k_position == 11
        elif q_position in {30}:
            return k_position == 32
        elif q_position in {32}:
            return k_position == 6
        elif q_position in {33}:
            return k_position == 35
        elif q_position in {34}:
            return k_position == 18
        elif q_position in {36}:
            return k_position == 27
        elif q_position in {37}:
            return k_position == 19
        elif q_position in {38}:
            return k_position == 37
        elif q_position in {39}:
            return k_position == 39

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0}:
            return k_position == 9
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {9, 2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 15
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 3
        elif q_position in {8, 21, 23}:
            return k_position == 24
        elif q_position in {17, 10}:
            return k_position == 11
        elif q_position in {27, 11}:
            return k_position == 12
        elif q_position in {12}:
            return k_position == 5
        elif q_position in {13}:
            return k_position == 14
        elif q_position in {16, 25, 14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 19
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {20}:
            return k_position == 25
        elif q_position in {26, 22}:
            return k_position == 8
        elif q_position in {24}:
            return k_position == 17
        elif q_position in {28}:
            return k_position == 23
        elif q_position in {29, 38}:
            return k_position == 20
        elif q_position in {32, 33, 34, 36, 30}:
            return k_position == 0
        elif q_position in {35, 31}:
            return k_position == 39
        elif q_position in {37}:
            return k_position == 34
        elif q_position in {39}:
            return k_position == 36

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(position, token):
        if position in {0, 1, 2, 12, 17, 29}:
            return token == "0"
        elif position in {3}:
            return token == "2"
        elif position in {4, 5, 7, 8, 11, 13, 23, 27}:
            return token == "3"
        elif position in {32, 6, 9, 10, 14, 15, 16, 20, 22, 24}:
            return token == "4"
        elif position in {33, 34, 35, 36, 37, 38, 39, 18, 26, 28, 30, 31}:
            return token == ""
        elif position in {25, 19, 21}:
            return token == "1"

    attn_0_5_pattern = select_closest(tokens, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(position, token):
        if position in {0, 33, 36, 15, 17, 29}:
            return token == "1"
        elif position in {1, 9, 11, 13, 14, 16, 19, 21, 24}:
            return token == "0"
        elif position in {2, 4, 6, 12, 20, 27}:
            return token == "3"
        elif position in {3, 5, 7}:
            return token == "2"
        elif position in {8, 18, 23, 25, 26}:
            return token == "4"
        elif position in {32, 34, 35, 10, 30, 31}:
            return token == ""
        elif position in {38, 22}:
            return token == "</s>"
        elif position in {28, 39}:
            return token == "<s>"
        elif position in {37}:
            return token == "<pad>"

    attn_0_6_pattern = select_closest(tokens, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0, 5, 39, 28, 29}:
            return k_position == 2
        elif q_position in {1, 3, 4, 15}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {6, 8, 11, 14, 18, 19, 20, 21, 24, 25}:
            return k_position == 5
        elif q_position in {7, 12, 13, 16, 22, 27}:
            return k_position == 4
        elif q_position in {9}:
            return k_position == 14
        elif q_position in {10}:
            return k_position == 12
        elif q_position in {17}:
            return k_position == 10
        elif q_position in {23}:
            return k_position == 11
        elif q_position in {26}:
            return k_position == 0
        elif q_position in {35, 30}:
            return k_position == 21
        elif q_position in {31}:
            return k_position == 39
        elif q_position in {32}:
            return k_position == 33
        elif q_position in {33}:
            return k_position == 26
        elif q_position in {34}:
            return k_position == 22
        elif q_position in {36}:
            return k_position == 17
        elif q_position in {37}:
            return k_position == 38
        elif q_position in {38}:
            return k_position == 32

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 1, 2, 3, 4, 5, 6, 33, 8, 9, 29, 30}:
            return token == ""
        elif position in {7}:
            return token == "</s>"
        elif position in {25, 10, 11}:
            return token == "<s>"
        elif position in {32, 34, 35, 36, 39, 12, 15, 17, 20, 22, 26, 31}:
            return token == "1"
        elif position in {13, 14, 16, 18, 19, 21, 23, 24, 27, 28}:
            return token == "0"
        elif position in {37, 38}:
            return token == "2"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {
            0,
            10,
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
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            30,
            34,
            37,
            39,
        }:
            return token == ""
        elif position in {1, 9}:
            return token == "</s>"
        elif position in {2, 7}:
            return token == "1"
        elif position in {32, 33, 3, 4, 5, 6, 35, 8, 36, 38, 29, 31}:
            return token == "0"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 32, 2, 34, 36, 37, 38, 7, 39, 29, 30, 31}:
            return token == "0"
        elif position in {1, 3, 4, 5, 6}:
            return token == "1"
        elif position in {
            8,
            9,
            10,
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
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            33,
            35,
        }:
            return token == ""

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 29}:
            return token == "1"
        elif position in {1}:
            return token == "<s>"
        elif position in {32, 33, 2, 3, 4, 5, 6, 7, 34, 36, 38, 39, 30, 31}:
            return token == "0"
        elif position in {8}:
            return token == "</s>"
        elif position in {
            9,
            10,
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
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            35,
            37,
        }:
            return token == ""

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(q_position, k_position):
        if q_position in {0, 34, 35, 38, 30, 31}:
            return k_position == 13
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {3, 21}:
            return k_position == 9
        elif q_position in {4}:
            return k_position == 10
        elif q_position in {5}:
            return k_position == 11
        elif q_position in {6, 39}:
            return k_position == 12
        elif q_position in {33, 7}:
            return k_position == 14
        elif q_position in {8, 9}:
            return k_position == 17
        elif q_position in {10, 11}:
            return k_position == 22
        elif q_position in {12}:
            return k_position == 21
        elif q_position in {16, 17, 13}:
            return k_position == 26
        elif q_position in {14, 15}:
            return k_position == 25
        elif q_position in {18}:
            return k_position == 8
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {20, 36}:
            return k_position == 16
        elif q_position in {24, 25, 22, 23}:
            return k_position == 5
        elif q_position in {26}:
            return k_position == 1
        elif q_position in {27, 28}:
            return k_position == 2
        elif q_position in {29}:
            return k_position == 15
        elif q_position in {32}:
            return k_position == 31
        elif q_position in {37}:
            return k_position == 20

    num_attn_0_4_pattern = select(positions, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(q_position, k_position):
        if q_position in {0}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {3}:
            return k_position == 8
        elif q_position in {4, 5}:
            return k_position == 9
        elif q_position in {6}:
            return k_position == 10
        elif q_position in {7}:
            return k_position == 11
        elif q_position in {8}:
            return k_position == 12
        elif q_position in {9}:
            return k_position == 14
        elif q_position in {10}:
            return k_position == 16
        elif q_position in {11, 12, 29}:
            return k_position == 18
        elif q_position in {33, 37, 13}:
            return k_position == 21
        elif q_position in {14}:
            return k_position == 20
        elif q_position in {15}:
            return k_position == 22
        elif q_position in {16}:
            return k_position == 23
        elif q_position in {17}:
            return k_position == 24
        elif q_position in {18, 19}:
            return k_position == 26
        elif q_position in {35, 20}:
            return k_position == 28
        elif q_position in {21}:
            return k_position == 35
        elif q_position in {28, 22}:
            return k_position == 34
        elif q_position in {23}:
            return k_position == 33
        elif q_position in {24}:
            return k_position == 30
        elif q_position in {25}:
            return k_position == 36
        elif q_position in {26}:
            return k_position == 32
        elif q_position in {27}:
            return k_position == 39
        elif q_position in {38, 34, 36, 30}:
            return k_position == 3
        elif q_position in {31}:
            return k_position == 27
        elif q_position in {32}:
            return k_position == 29
        elif q_position in {39}:
            return k_position == 17

    num_attn_0_5_pattern = select(positions, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(q_position, k_position):
        if q_position in {0, 11}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 37
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {4, 5}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 10
        elif q_position in {9}:
            return k_position == 12
        elif q_position in {10}:
            return k_position == 13
        elif q_position in {12, 38}:
            return k_position == 16
        elif q_position in {13, 14}:
            return k_position == 18
        elif q_position in {15}:
            return k_position == 19
        elif q_position in {16}:
            return k_position == 20
        elif q_position in {17}:
            return k_position == 22
        elif q_position in {18, 31}:
            return k_position == 21
        elif q_position in {19}:
            return k_position == 24
        elif q_position in {20}:
            return k_position == 25
        elif q_position in {36, 21}:
            return k_position == 26
        elif q_position in {22}:
            return k_position == 27
        elif q_position in {24, 37, 23}:
            return k_position == 28
        elif q_position in {25}:
            return k_position == 36
        elif q_position in {26, 34, 30}:
            return k_position == 34
        elif q_position in {27}:
            return k_position == 35
        elif q_position in {28}:
            return k_position == 38
        elif q_position in {29}:
            return k_position == 23
        elif q_position in {32}:
            return k_position == 31
        elif q_position in {33}:
            return k_position == 29
        elif q_position in {35, 39}:
            return k_position == 17

    num_attn_0_6_pattern = select(positions, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(q_position, k_position):
        if q_position in {0, 12}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 21
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 8
        elif q_position in {6}:
            return k_position == 9
        elif q_position in {29, 7}:
            return k_position == 10
        elif q_position in {8, 9}:
            return k_position == 11
        elif q_position in {10}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 14
        elif q_position in {13, 14}:
            return k_position == 16
        elif q_position in {15}:
            return k_position == 18
        elif q_position in {16}:
            return k_position == 19
        elif q_position in {17}:
            return k_position == 20
        elif q_position in {18, 20}:
            return k_position == 23
        elif q_position in {19, 37}:
            return k_position == 22
        elif q_position in {21}:
            return k_position == 24
        elif q_position in {22, 39}:
            return k_position == 25
        elif q_position in {23}:
            return k_position == 26
        elif q_position in {24}:
            return k_position == 27
        elif q_position in {25}:
            return k_position == 28
        elif q_position in {26}:
            return k_position == 35
        elif q_position in {32, 34, 27}:
            return k_position == 38
        elif q_position in {28}:
            return k_position == 32
        elif q_position in {30}:
            return k_position == 37
        elif q_position in {35, 31}:
            return k_position == 2
        elif q_position in {33}:
            return k_position == 34
        elif q_position in {36}:
            return k_position == 30
        elif q_position in {38}:
            return k_position == 31

    num_attn_0_7_pattern = select(positions, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position):
        key = position
        if key in {1, 3, 29}:
            return 16
        elif key in {4}:
            return 2
        return 27

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in positions]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_1_output, attn_0_5_output):
        key = (attn_0_1_output, attn_0_5_output)
        return 4

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_5_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(position):
        key = position
        if key in {1, 2, 3, 4, 5, 29, 31, 32, 34, 36, 39}:
            return 11
        elif key in {0, 6, 7, 8, 9, 10, 11}:
            return 38
        elif key in {12}:
            return 14
        elif key in {13}:
            return 29
        return 35

    mlp_0_2_outputs = [mlp_0_2(k0) for k0 in positions]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(position):
        key = position
        if key in {3, 4, 5, 6, 8}:
            return 33
        elif key in {1, 2, 29}:
            return 37
        return 17

    mlp_0_3_outputs = [mlp_0_3(k0) for k0 in positions]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output, num_attn_0_4_output):
        key = (num_attn_0_2_output, num_attn_0_4_output)
        return 30

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_4_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_4_output):
        key = num_attn_0_4_output
        return 16

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_4_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_2_output, num_attn_0_6_output):
        key = (num_attn_0_2_output, num_attn_0_6_output)
        return 15

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_1_output, num_attn_0_5_output):
        key = (num_attn_0_1_output, num_attn_0_5_output)
        return 38

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_5_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_4_output, num_mlp_0_0_output):
        if attn_0_4_output in {"0"}:
            return num_mlp_0_0_output == 29
        elif attn_0_4_output in {"1"}:
            return num_mlp_0_0_output == 30
        elif attn_0_4_output in {"2"}:
            return num_mlp_0_0_output == 12
        elif attn_0_4_output in {"3"}:
            return num_mlp_0_0_output == 32
        elif attn_0_4_output in {"4"}:
            return num_mlp_0_0_output == 38
        elif attn_0_4_output in {"</s>"}:
            return num_mlp_0_0_output == 2
        elif attn_0_4_output in {"<s>"}:
            return num_mlp_0_0_output == 6

    attn_1_0_pattern = select_closest(
        num_mlp_0_0_outputs, attn_0_4_outputs, predicate_1_0
    )
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(position, mlp_0_3_output):
        if position in {0, 8}:
            return mlp_0_3_output == 37
        elif position in {1, 3, 37, 31}:
            return mlp_0_3_output == 1
        elif position in {2, 13, 14, 23, 24}:
            return mlp_0_3_output == 33
        elif position in {34, 4}:
            return mlp_0_3_output == 18
        elif position in {5, 6, 39, 9, 20}:
            return mlp_0_3_output == 17
        elif position in {27, 30, 7}:
            return mlp_0_3_output == 5
        elif position in {10}:
            return mlp_0_3_output == 14
        elif position in {11}:
            return mlp_0_3_output == 16
        elif position in {12, 15}:
            return mlp_0_3_output == 10
        elif position in {16, 38}:
            return mlp_0_3_output == 27
        elif position in {17, 28, 29}:
            return mlp_0_3_output == 2
        elif position in {18}:
            return mlp_0_3_output == 38
        elif position in {19}:
            return mlp_0_3_output == 9
        elif position in {21}:
            return mlp_0_3_output == 11
        elif position in {33, 22}:
            return mlp_0_3_output == 8
        elif position in {25, 26, 35}:
            return mlp_0_3_output == 4
        elif position in {32}:
            return mlp_0_3_output == 34
        elif position in {36}:
            return mlp_0_3_output == 25

    attn_1_1_pattern = select_closest(mlp_0_3_outputs, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_4_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(mlp_0_3_output, token):
        if mlp_0_3_output in {0, 6}:
            return token == "<s>"
        elif mlp_0_3_output in {1, 36, 38, 39, 17, 20, 22, 24, 27, 28}:
            return token == "4"
        elif mlp_0_3_output in {32, 2, 35, 34, 29}:
            return token == "3"
        elif mlp_0_3_output in {3, 5, 7, 8, 9, 10, 11, 16, 18, 19, 23, 25, 30}:
            return token == ""
        elif mlp_0_3_output in {4}:
            return token == "</s>"
        elif mlp_0_3_output in {37, 12, 13, 31}:
            return token == "0"
        elif mlp_0_3_output in {21, 14}:
            return token == "1"
        elif mlp_0_3_output in {33, 26, 15}:
            return token == "2"

    attn_1_2_pattern = select_closest(tokens, mlp_0_3_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(mlp_0_2_output, token):
        if mlp_0_2_output in {0, 4, 5, 39, 12, 13, 16, 17, 18, 21, 24, 25, 26, 27}:
            return token == ""
        elif mlp_0_2_output in {1, 2, 3, 35, 36, 6, 10, 14, 15, 19, 23, 28, 31}:
            return token == "4"
        elif mlp_0_2_output in {34, 38, 7, 8, 20, 22, 29}:
            return token == "3"
        elif mlp_0_2_output in {9, 37}:
            return token == "0"
        elif mlp_0_2_output in {11}:
            return token == "<s>"
        elif mlp_0_2_output in {30}:
            return token == "1"
        elif mlp_0_2_output in {32, 33}:
            return token == "2"

    attn_1_3_pattern = select_closest(tokens, mlp_0_2_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(position, token):
        if position in {0, 35, 5, 7, 10, 11, 20, 23}:
            return token == "3"
        elif position in {32, 1, 3, 37, 6, 8, 9, 14, 16, 17, 18, 28, 30, 31}:
            return token == ""
        elif position in {2}:
            return token == "2"
        elif position in {4, 12, 21, 22, 25}:
            return token == "4"
        elif position in {24, 19, 13}:
            return token == "0"
        elif position in {36, 39, 38, 15}:
            return token == "<pad>"
        elif position in {26, 27}:
            return token == "</s>"
        elif position in {29}:
            return token == "1"
        elif position in {33, 34}:
            return token == "<s>"

    attn_1_4_pattern = select_closest(tokens, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, tokens)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(q_mlp_0_3_output, k_mlp_0_3_output):
        if q_mlp_0_3_output in {0, 32, 34, 5, 38, 30}:
            return k_mlp_0_3_output == 3
        elif q_mlp_0_3_output in {1, 22, 9, 17}:
            return k_mlp_0_3_output == 37
        elif q_mlp_0_3_output in {2}:
            return k_mlp_0_3_output == 4
        elif q_mlp_0_3_output in {35, 3, 13, 39}:
            return k_mlp_0_3_output == 33
        elif q_mlp_0_3_output in {4, 36, 7}:
            return k_mlp_0_3_output == 6
        elif q_mlp_0_3_output in {25, 33, 6, 23}:
            return k_mlp_0_3_output == 7
        elif q_mlp_0_3_output in {8, 26, 27}:
            return k_mlp_0_3_output == 5
        elif q_mlp_0_3_output in {10, 18, 31}:
            return k_mlp_0_3_output == 15
        elif q_mlp_0_3_output in {11}:
            return k_mlp_0_3_output == 34
        elif q_mlp_0_3_output in {12}:
            return k_mlp_0_3_output == 17
        elif q_mlp_0_3_output in {14}:
            return k_mlp_0_3_output == 16
        elif q_mlp_0_3_output in {15}:
            return k_mlp_0_3_output == 19
        elif q_mlp_0_3_output in {16}:
            return k_mlp_0_3_output == 9
        elif q_mlp_0_3_output in {19, 28, 37}:
            return k_mlp_0_3_output == 1
        elif q_mlp_0_3_output in {20, 29}:
            return k_mlp_0_3_output == 20
        elif q_mlp_0_3_output in {21}:
            return k_mlp_0_3_output == 38
        elif q_mlp_0_3_output in {24}:
            return k_mlp_0_3_output == 8

    attn_1_5_pattern = select_closest(mlp_0_3_outputs, mlp_0_3_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_5_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(position, token):
        if position in {0, 34, 35, 4, 5, 6, 38, 10, 12, 15, 16, 18, 22, 24, 25, 26}:
            return token == "4"
        elif position in {1, 20, 9, 17}:
            return token == "0"
        elif position in {32, 33, 2, 3, 36, 37, 7, 8, 11, 19, 28, 29, 30}:
            return token == ""
        elif position in {39, 13, 21, 23, 31}:
            return token == "3"
        elif position in {27, 14}:
            return token == "1"

    attn_1_6_pattern = select_closest(tokens, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, tokens)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(attn_0_3_output, token):
        if attn_0_3_output in {"2", "1", "0"}:
            return token == "3"
        elif attn_0_3_output in {"3", "<s>"}:
            return token == ""
        elif attn_0_3_output in {"4"}:
            return token == "4"
        elif attn_0_3_output in {"</s>"}:
            return token == "</s>"

    attn_1_7_pattern = select_closest(tokens, attn_0_3_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, tokens)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, attn_0_6_output):
        if position in {0, 35, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28}:
            return attn_0_6_output == ""
        elif position in {1, 3}:
            return attn_0_6_output == "3"
        elif position in {32, 2, 34, 4, 5, 36, 39, 8, 9, 10, 29, 30, 31}:
            return attn_0_6_output == "0"
        elif position in {38, 33, 37, 6}:
            return attn_0_6_output == "1"
        elif position in {7}:
            return attn_0_6_output == "2"
        elif position in {17, 18, 11, 12}:
            return attn_0_6_output == "<s>"
        elif position in {16, 14, 15}:
            return attn_0_6_output == "</s>"

    num_attn_1_0_pattern = select(attn_0_6_outputs, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_0_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 20, 29}:
            return token == "1"
        elif mlp_0_0_output in {1}:
            return token == "</s>"
        elif mlp_0_0_output in {
            2,
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
            14,
            15,
            17,
            18,
            19,
            21,
            22,
            23,
            24,
            26,
            27,
            28,
            30,
            32,
            33,
            34,
            35,
            39,
        }:
            return token == ""
        elif mlp_0_0_output in {3, 37, 38, 31}:
            return token == "0"
        elif mlp_0_0_output in {16, 36}:
            return token == "2"
        elif mlp_0_0_output in {25}:
            return token == "<s>"

    num_attn_1_1_pattern = select(tokens, mlp_0_0_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_3_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(mlp_0_2_output, position):
        if mlp_0_2_output in {0}:
            return position == 17
        elif mlp_0_2_output in {1, 19, 4, 37}:
            return position == 1
        elif mlp_0_2_output in {25, 2, 3}:
            return position == 11
        elif mlp_0_2_output in {5}:
            return position == 21
        elif mlp_0_2_output in {6}:
            return position == 10
        elif mlp_0_2_output in {8, 36, 7}:
            return position == 37
        elif mlp_0_2_output in {9}:
            return position == 24
        elif mlp_0_2_output in {10}:
            return position == 13
        elif mlp_0_2_output in {11, 15, 20, 22, 24}:
            return position == 2
        elif mlp_0_2_output in {12}:
            return position == 28
        elif mlp_0_2_output in {13, 39}:
            return position == 27
        elif mlp_0_2_output in {34, 14}:
            return position == 12
        elif mlp_0_2_output in {16}:
            return position == 3
        elif mlp_0_2_output in {32, 17, 23}:
            return position == 7
        elif mlp_0_2_output in {18}:
            return position == 9
        elif mlp_0_2_output in {21}:
            return position == 6
        elif mlp_0_2_output in {26}:
            return position == 39
        elif mlp_0_2_output in {27}:
            return position == 8
        elif mlp_0_2_output in {28}:
            return position == 16
        elif mlp_0_2_output in {35, 29, 31}:
            return position == 35
        elif mlp_0_2_output in {30}:
            return position == 15
        elif mlp_0_2_output in {33}:
            return position == 20
        elif mlp_0_2_output in {38}:
            return position == 31

    num_attn_1_2_pattern = select(positions, mlp_0_2_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_1_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, token):
        if position in {0}:
            return token == "</s>"
        elif position in {1, 21, 23, 26, 27}:
            return token == "1"
        elif position in {25, 2, 22}:
            return token == "0"
        elif position in {
            3,
            4,
            5,
            6,
            7,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            29,
            33,
            36,
            37,
            38,
        }:
            return token == ""
        elif position in {32, 34, 35, 39, 8, 30, 31}:
            return token == "<pad>"
        elif position in {19}:
            return token == "<s>"
        elif position in {20}:
            return token == "3"
        elif position in {24, 28}:
            return token == "2"

    num_attn_1_3_pattern = select(tokens, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_3_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(position, attn_0_1_output):
        if position in {0, 29}:
            return attn_0_1_output == "1"
        elif position in {1, 4}:
            return attn_0_1_output == "3"
        elif position in {32, 33, 2, 3, 5, 6, 7, 8, 39, 10, 11, 30}:
            return attn_0_1_output == "2"
        elif position in {9, 37, 31}:
            return attn_0_1_output == "0"
        elif position in {
            35,
            36,
            38,
            12,
            13,
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
        }:
            return attn_0_1_output == ""
        elif position in {16, 14}:
            return attn_0_1_output == "</s>"
        elif position in {15}:
            return attn_0_1_output == "<s>"
        elif position in {34}:
            return attn_0_1_output == "4"

    num_attn_1_4_pattern = select(attn_0_1_outputs, positions, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_0_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(position, token):
        if position in {0, 33, 32, 36, 38}:
            return token == "1"
        elif position in {1}:
            return token == "3"
        elif position in {2, 4, 6, 39, 30}:
            return token == "2"
        elif position in {16, 3, 14}:
            return token == "<s>"
        elif position in {
            34,
            35,
            5,
            7,
            12,
            13,
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
        }:
            return token == ""
        elif position in {8, 9, 10, 11, 15, 17, 29}:
            return token == "0"
        elif position in {37, 31}:
            return token == "</s>"

    num_attn_1_5_pattern = select(tokens, positions, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_3_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(position, attn_0_6_output):
        if position in {0, 35, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return attn_0_6_output == ""
        elif position in {1}:
            return attn_0_6_output == "4"
        elif position in {2, 4}:
            return attn_0_6_output == "2"
        elif position in {32, 33, 34, 3, 36, 5, 6, 7, 8, 9, 10, 38, 12, 39, 14, 30, 31}:
            return attn_0_6_output == "1"
        elif position in {11}:
            return attn_0_6_output == "0"
        elif position in {37, 13, 15}:
            return attn_0_6_output == "<s>"
        elif position in {16}:
            return attn_0_6_output == "</s>"

    num_attn_1_6_pattern = select(attn_0_6_outputs, positions, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_0_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(attn_0_3_output, token):
        if attn_0_3_output in {"2", "1", "3", "0", "</s>", "4", "<s>"}:
            return token == "2"

    num_attn_1_7_pattern = select(tokens, attn_0_3_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, ones)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_5_output, attn_1_0_output):
        key = (attn_1_5_output, attn_1_0_output)
        if key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "3"),
            ("0", "4"),
            ("0", "</s>"),
            ("0", "<s>"),
            ("1", "0"),
            ("1", "1"),
            ("1", "3"),
            ("1", "4"),
            ("1", "</s>"),
            ("1", "<s>"),
            ("2", "</s>"),
            ("2", "<s>"),
            ("3", "0"),
            ("3", "1"),
            ("3", "3"),
            ("3", "4"),
            ("3", "</s>"),
            ("3", "<s>"),
            ("4", "0"),
            ("4", "1"),
            ("4", "3"),
            ("4", "4"),
            ("4", "</s>"),
            ("4", "<s>"),
            ("</s>", "0"),
            ("</s>", "1"),
            ("</s>", "3"),
            ("</s>", "4"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
            ("<s>", "0"),
            ("<s>", "1"),
            ("<s>", "2"),
            ("<s>", "3"),
            ("<s>", "4"),
            ("<s>", "</s>"),
            ("<s>", "<s>"),
        }:
            return 38
        return 39

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_5_outputs, attn_1_0_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_5_output, attn_1_0_output):
        key = (attn_1_5_output, attn_1_0_output)
        if key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "2"),
            ("0", "3"),
            ("0", "</s>"),
            ("0", "<s>"),
            ("1", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("1", "<s>"),
            ("2", "<s>"),
            ("3", "0"),
            ("3", "1"),
            ("3", "2"),
            ("3", "3"),
            ("3", "<s>"),
            ("</s>", "0"),
            ("</s>", "1"),
            ("</s>", "2"),
            ("</s>", "3"),
            ("</s>", "<s>"),
            ("<s>", "<s>"),
        }:
            return 0
        elif key in {("1", "4"), ("1", "</s>")}:
            return 18
        return 4

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_5_outputs, attn_1_0_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_0_1_output, position):
        key = (attn_0_1_output, position)
        if key in {
            ("0", 0),
            ("0", 2),
            ("0", 7),
            ("0", 13),
            ("0", 18),
            ("0", 19),
            ("0", 22),
            ("0", 23),
            ("0", 24),
            ("0", 27),
            ("0", 28),
            ("0", 36),
            ("0", 38),
            ("0", 39),
            ("1", 0),
            ("1", 2),
            ("1", 7),
            ("1", 13),
            ("1", 18),
            ("1", 19),
            ("1", 22),
            ("1", 23),
            ("1", 24),
            ("1", 27),
            ("1", 39),
            ("2", 0),
            ("2", 2),
            ("2", 7),
            ("2", 13),
            ("2", 18),
            ("2", 19),
            ("2", 22),
            ("2", 23),
            ("2", 24),
            ("2", 27),
            ("2", 28),
            ("2", 38),
            ("2", 39),
            ("3", 0),
            ("3", 2),
            ("3", 7),
            ("3", 13),
            ("3", 18),
            ("3", 19),
            ("3", 22),
            ("3", 23),
            ("3", 24),
            ("3", 27),
            ("3", 28),
            ("3", 38),
            ("3", 39),
            ("4", 0),
            ("4", 2),
            ("4", 7),
            ("4", 13),
            ("4", 18),
            ("4", 19),
            ("4", 22),
            ("4", 23),
            ("4", 24),
            ("4", 27),
            ("4", 28),
            ("4", 38),
            ("4", 39),
            ("</s>", 0),
            ("</s>", 2),
            ("</s>", 7),
            ("</s>", 13),
            ("</s>", 18),
            ("</s>", 19),
            ("</s>", 22),
            ("</s>", 23),
            ("</s>", 24),
            ("</s>", 27),
            ("</s>", 39),
            ("<s>", 0),
            ("<s>", 2),
            ("<s>", 7),
            ("<s>", 13),
            ("<s>", 18),
            ("<s>", 19),
            ("<s>", 22),
            ("<s>", 23),
            ("<s>", 24),
            ("<s>", 27),
            ("<s>", 28),
            ("<s>", 38),
            ("<s>", 39),
        }:
            return 11
        return 17

    mlp_1_2_outputs = [mlp_1_2(k0, k1) for k0, k1 in zip(attn_0_1_outputs, positions)]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(mlp_0_3_output, attn_0_2_output):
        key = (mlp_0_3_output, attn_0_2_output)
        return 29

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(mlp_0_3_outputs, attn_0_2_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_3_output, num_attn_0_4_output):
        key = (num_attn_1_3_output, num_attn_0_4_output)
        if key in {(0, 0), (1, 0), (2, 0)}:
            return 8
        return 26

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_0_4_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_0_output, num_attn_1_3_output):
        key = (num_attn_1_0_output, num_attn_1_3_output)
        return 16

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_6_output, num_attn_1_2_output):
        key = (num_attn_1_6_output, num_attn_1_2_output)
        return 29

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_0_5_output):
        key = num_attn_0_5_output
        if key in {0}:
            return 27
        return 16

    num_mlp_1_3_outputs = [num_mlp_1_3(k0) for k0 in num_attn_0_5_outputs]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_token, k_token):
        if q_token in {"2", "1", "0"}:
            return k_token == "4"
        elif q_token in {"3", "<s>"}:
            return k_token == "</s>"
        elif q_token in {"4"}:
            return k_token == "3"
        elif q_token in {"</s>"}:
            return k_token == "<s>"

    attn_2_0_pattern = select_closest(tokens, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_7_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(mlp_0_0_output, attn_1_7_output):
        if mlp_0_0_output in {
            0,
            1,
            4,
            8,
            9,
            10,
            12,
            13,
            18,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            28,
            31,
            34,
            35,
            37,
            39,
        }:
            return attn_1_7_output == ""
        elif mlp_0_0_output in {33, 2, 38, 17, 19, 29}:
            return attn_1_7_output == "</s>"
        elif mlp_0_0_output in {3}:
            return attn_1_7_output == "4"
        elif mlp_0_0_output in {27, 5, 6}:
            return attn_1_7_output == "<s>"
        elif mlp_0_0_output in {16, 11, 15, 7}:
            return attn_1_7_output == "2"
        elif mlp_0_0_output in {14}:
            return attn_1_7_output == "3"
        elif mlp_0_0_output in {30}:
            return attn_1_7_output == "<pad>"
        elif mlp_0_0_output in {32}:
            return attn_1_7_output == "1"
        elif mlp_0_0_output in {36}:
            return attn_1_7_output == "0"

    attn_2_1_pattern = select_closest(attn_1_7_outputs, mlp_0_0_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_attn_1_0_output, k_attn_1_0_output):
        if q_attn_1_0_output in {"0"}:
            return k_attn_1_0_output == "2"
        elif q_attn_1_0_output in {"1"}:
            return k_attn_1_0_output == "0"
        elif q_attn_1_0_output in {"2", "<s>", "3"}:
            return k_attn_1_0_output == ""
        elif q_attn_1_0_output in {"4"}:
            return k_attn_1_0_output == "</s>"
        elif q_attn_1_0_output in {"</s>"}:
            return k_attn_1_0_output == "<pad>"

    attn_2_2_pattern = select_closest(attn_1_0_outputs, attn_1_0_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_7_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_2_output, mlp_0_0_output):
        if attn_0_2_output in {"0"}:
            return mlp_0_0_output == 4
        elif attn_0_2_output in {"1"}:
            return mlp_0_0_output == 7
        elif attn_0_2_output in {"2"}:
            return mlp_0_0_output == 27
        elif attn_0_2_output in {"3"}:
            return mlp_0_0_output == 3
        elif attn_0_2_output in {"4"}:
            return mlp_0_0_output == 2
        elif attn_0_2_output in {"<s>", "</s>"}:
            return mlp_0_0_output == 6

    attn_2_3_pattern = select_closest(mlp_0_0_outputs, attn_0_2_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_7_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(mlp_0_3_output, attn_1_2_output):
        if mlp_0_3_output in {0, 35, 22}:
            return attn_1_2_output == "3"
        elif mlp_0_3_output in {
            1,
            2,
            3,
            4,
            5,
            8,
            12,
            14,
            15,
            16,
            19,
            21,
            23,
            24,
            26,
            29,
            31,
            32,
            34,
            36,
        }:
            return attn_1_2_output == ""
        elif mlp_0_3_output in {25, 28, 6, 30}:
            return attn_1_2_output == "</s>"
        elif mlp_0_3_output in {18, 38, 7}:
            return attn_1_2_output == "2"
        elif mlp_0_3_output in {9, 17, 39}:
            return attn_1_2_output == "1"
        elif mlp_0_3_output in {10}:
            return attn_1_2_output == "<pad>"
        elif mlp_0_3_output in {27, 11, 37}:
            return attn_1_2_output == "0"
        elif mlp_0_3_output in {33, 13}:
            return attn_1_2_output == "<s>"
        elif mlp_0_3_output in {20}:
            return attn_1_2_output == "4"

    attn_2_4_pattern = select_closest(attn_1_2_outputs, mlp_0_3_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_0_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(mlp_0_3_output, attn_1_2_output):
        if mlp_0_3_output in {0, 35, 4, 6, 7, 8, 11, 17, 30}:
            return attn_1_2_output == "<s>"
        elif mlp_0_3_output in {1, 2, 5, 12, 31}:
            return attn_1_2_output == "3"
        elif mlp_0_3_output in {3}:
            return attn_1_2_output == "<pad>"
        elif mlp_0_3_output in {32, 9, 21, 14}:
            return attn_1_2_output == "4"
        elif mlp_0_3_output in {
            33,
            34,
            36,
            38,
            10,
            13,
            15,
            16,
            19,
            20,
            22,
            24,
            25,
            28,
            29,
        }:
            return attn_1_2_output == ""
        elif mlp_0_3_output in {18, 27, 39}:
            return attn_1_2_output == "</s>"
        elif mlp_0_3_output in {26, 37, 23}:
            return attn_1_2_output == "2"

    attn_2_5_pattern = select_closest(attn_1_2_outputs, mlp_0_3_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_1_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(q_position, k_position):
        if q_position in {0, 33, 36, 6, 27}:
            return k_position == 0
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {9, 10, 3}:
            return k_position == 15
        elif q_position in {4}:
            return k_position == 9
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {24, 20, 7}:
            return k_position == 25
        elif q_position in {8, 32}:
            return k_position == 14
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 28
        elif q_position in {28, 13}:
            return k_position == 5
        elif q_position in {17, 14, 22}:
            return k_position == 21
        elif q_position in {16, 31, 15}:
            return k_position == 18
        elif q_position in {18, 19}:
            return k_position == 29
        elif q_position in {21}:
            return k_position == 23
        elif q_position in {23}:
            return k_position == 26
        elif q_position in {25}:
            return k_position == 24
        elif q_position in {26}:
            return k_position == 20
        elif q_position in {29}:
            return k_position == 17
        elif q_position in {30}:
            return k_position == 32
        elif q_position in {34, 37, 39}:
            return k_position == 3
        elif q_position in {35}:
            return k_position == 36
        elif q_position in {38}:
            return k_position == 11

    attn_2_6_pattern = select_closest(positions, positions, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_6_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_1_1_output, mlp_0_3_output):
        if attn_1_1_output in {"4", "0"}:
            return mlp_0_3_output == 33
        elif attn_1_1_output in {"1"}:
            return mlp_0_3_output == 11
        elif attn_1_1_output in {"2"}:
            return mlp_0_3_output == 37
        elif attn_1_1_output in {"3"}:
            return mlp_0_3_output == 20
        elif attn_1_1_output in {"</s>"}:
            return mlp_0_3_output == 7
        elif attn_1_1_output in {"<s>"}:
            return mlp_0_3_output == 6

    attn_2_7_pattern = select_closest(mlp_0_3_outputs, attn_1_1_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_7_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_0_2_output, token):
        if mlp_0_2_output in {0, 26, 39}:
            return token == "</s>"
        elif mlp_0_2_output in {
            1,
            2,
            3,
            7,
            8,
            10,
            12,
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
            27,
            28,
            29,
            30,
            31,
            33,
            34,
            35,
            36,
            38,
        }:
            return token == ""
        elif mlp_0_2_output in {32, 9, 4, 13}:
            return token == "1"
        elif mlp_0_2_output in {11, 5, 6}:
            return token == "2"
        elif mlp_0_2_output in {37, 14}:
            return token == "<s>"

    num_attn_2_0_pattern = select(tokens, mlp_0_2_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(mlp_0_2_output, attn_1_4_output):
        if mlp_0_2_output in {
            0,
            1,
            2,
            3,
            4,
            6,
            7,
            8,
            10,
            11,
            12,
            14,
            15,
            16,
            18,
            19,
            20,
            21,
            25,
            27,
            29,
            30,
            31,
            32,
            33,
            36,
            37,
            38,
            39,
        }:
            return attn_1_4_output == "0"
        elif mlp_0_2_output in {35, 5, 9, 17, 23, 28}:
            return attn_1_4_output == ""
        elif mlp_0_2_output in {34, 13, 22}:
            return attn_1_4_output == "</s>"
        elif mlp_0_2_output in {24}:
            return attn_1_4_output == "2"
        elif mlp_0_2_output in {26}:
            return attn_1_4_output == "1"

    num_attn_2_1_pattern = select(attn_1_4_outputs, mlp_0_2_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_0_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_5_output, position):
        if attn_1_5_output in {"0"}:
            return position == 1
        elif attn_1_5_output in {"1"}:
            return position == 20
        elif attn_1_5_output in {"2"}:
            return position == 0
        elif attn_1_5_output in {"3", "4"}:
            return position == 21
        elif attn_1_5_output in {"<s>", "</s>"}:
            return position == 2

    num_attn_2_2_pattern = select(positions, attn_1_5_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_3_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(mlp_0_2_output, token):
        if mlp_0_2_output in {
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            10,
            11,
            12,
            15,
            16,
            18,
            19,
            21,
            23,
            24,
            25,
            28,
            29,
            30,
            31,
            33,
            34,
            36,
            37,
            38,
            39,
        }:
            return token == "1"
        elif mlp_0_2_output in {32, 35, 9, 13, 14, 17, 20, 22, 26, 27}:
            return token == ""

    num_attn_2_3_pattern = select(tokens, mlp_0_2_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, ones)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 1, 33, 36, 37, 18, 20, 23, 25, 28, 30}:
            return token == "3"
        elif mlp_0_0_output in {
            32,
            2,
            4,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            16,
            21,
            24,
            26,
            29,
        }:
            return token == "0"
        elif mlp_0_0_output in {3, 31}:
            return token == "<s>"
        elif mlp_0_0_output in {35, 27, 5}:
            return token == ""
        elif mlp_0_0_output in {15}:
            return token == "</s>"
        elif mlp_0_0_output in {17, 38}:
            return token == "4"
        elif mlp_0_0_output in {19}:
            return token == "1"
        elif mlp_0_0_output in {34, 22, 39}:
            return token == "2"

    num_attn_2_4_pattern = select(tokens, mlp_0_0_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_3_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_1_5_output, attn_0_3_output):
        if attn_1_5_output in {"<s>", "1", "0", "</s>"}:
            return attn_0_3_output == "1"
        elif attn_1_5_output in {"2"}:
            return attn_0_3_output == "0"
        elif attn_1_5_output in {"3", "4"}:
            return attn_0_3_output == ""

    num_attn_2_5_pattern = select(attn_0_3_outputs, attn_1_5_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_0_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(mlp_0_2_output, attn_1_6_output):
        if mlp_0_2_output in {
            0,
            4,
            6,
            7,
            8,
            9,
            10,
            12,
            15,
            19,
            20,
            21,
            25,
            27,
            28,
            30,
            31,
            33,
            35,
            36,
            37,
            38,
            39,
        }:
            return attn_1_6_output == ""
        elif mlp_0_2_output in {1, 2, 34, 5, 11, 13, 16, 18, 22, 23, 24, 26}:
            return attn_1_6_output == "0"
        elif mlp_0_2_output in {3, 29, 14}:
            return attn_1_6_output == "</s>"
        elif mlp_0_2_output in {17}:
            return attn_1_6_output == "1"
        elif mlp_0_2_output in {32}:
            return attn_1_6_output == "<s>"

    num_attn_2_6_pattern = select(attn_1_6_outputs, mlp_0_2_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_3_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_1_5_output, attn_0_6_output):
        if attn_1_5_output in {"2", "<s>", "4", "0"}:
            return attn_0_6_output == ""
        elif attn_1_5_output in {"1"}:
            return attn_0_6_output == "<s>"
        elif attn_1_5_output in {"3"}:
            return attn_0_6_output == "1"
        elif attn_1_5_output in {"</s>"}:
            return attn_0_6_output == "</s>"

    num_attn_2_7_pattern = select(attn_0_6_outputs, attn_1_5_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_3_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(num_mlp_0_0_output, num_mlp_1_1_output):
        key = (num_mlp_0_0_output, num_mlp_1_1_output)
        if key in {
            (8, 26),
            (10, 26),
            (11, 26),
            (12, 26),
            (13, 26),
            (14, 26),
            (15, 26),
            (17, 26),
            (21, 26),
            (24, 26),
            (29, 26),
            (30, 26),
            (31, 26),
            (32, 26),
            (35, 26),
            (36, 26),
            (39, 26),
        }:
            return 16
        elif key in {(1, 24), (9, 24), (9, 25), (10, 24), (34, 24), (34, 25), (39, 24)}:
            return 32
        return 39

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(num_mlp_0_0_outputs, num_mlp_1_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(num_mlp_0_3_output, attn_2_0_output):
        key = (num_mlp_0_3_output, attn_2_0_output)
        if key in {
            (6, "0"),
            (6, "1"),
            (6, "2"),
            (6, "3"),
            (6, "4"),
            (6, "<s>"),
            (23, "4"),
            (26, "0"),
            (26, "3"),
            (26, "4"),
            (26, "<s>"),
            (28, "0"),
            (28, "4"),
            (39, "0"),
            (39, "1"),
            (39, "2"),
            (39, "3"),
            (39, "4"),
            (39, "</s>"),
            (39, "<s>"),
        }:
            return 10
        return 27

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(num_mlp_0_3_outputs, attn_2_0_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(mlp_1_2_output, position):
        key = (mlp_1_2_output, position)
        if key in {
            (2, 14),
            (2, 24),
            (5, 2),
            (5, 8),
            (5, 13),
            (5, 14),
            (5, 16),
            (5, 22),
            (5, 24),
            (5, 27),
            (5, 32),
            (16, 14),
            (16, 24),
            (21, 14),
            (21, 24),
            (25, 14),
            (25, 24),
            (26, 14),
            (26, 24),
            (26, 27),
            (30, 14),
            (30, 24),
            (30, 27),
            (35, 14),
            (35, 24),
            (35, 27),
        }:
            return 24
        return 7

    mlp_2_2_outputs = [mlp_2_2(k0, k1) for k0, k1 in zip(mlp_1_2_outputs, positions)]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(num_mlp_1_3_output, attn_0_2_output):
        key = (num_mlp_1_3_output, attn_0_2_output)
        if key in {
            (19, "0"),
            (19, "1"),
            (19, "2"),
            (19, "3"),
            (19, "4"),
            (19, "</s>"),
            (19, "<s>"),
        }:
            return 10
        return 26

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(num_mlp_1_3_outputs, attn_0_2_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_1_output):
        key = num_attn_1_1_output
        return 11

    num_mlp_2_0_outputs = [num_mlp_2_0(k0) for k0 in num_attn_1_1_outputs]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_3_output, num_attn_1_0_output):
        key = (num_attn_1_3_output, num_attn_1_0_output)
        return 14

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_0_5_output, num_attn_0_7_output):
        key = (num_attn_0_5_output, num_attn_0_7_output)
        if key in {(0, 0)}:
            return 6
        return 0

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_0_5_outputs, num_attn_0_7_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_2_6_output, num_attn_2_0_output):
        key = (num_attn_2_6_output, num_attn_2_0_output)
        return 30

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_2_6_outputs, num_attn_2_0_outputs)
    ]
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


print(run(["<s>", "0", "1", "3", "0", "0", "0", "3", "2", "3", "1", "1", "</s>"]))
