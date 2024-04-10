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
        "output/length/rasp/sort/trainlength30/s2/sort_weights.csv",
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
        if position in {0, 1, 23, 24, 25}:
            return token == "1"
        elif position in {2, 4}:
            return token == "0"
        elif position in {3, 5}:
            return token == "2"
        elif position in {6, 7, 8, 10, 12, 13, 16, 21, 22, 26, 28, 29}:
            return token == "3"
        elif position in {9, 11, 14, 17, 18, 19, 20}:
            return token == "4"
        elif position in {32, 33, 34, 35, 36, 37, 38, 39, 15, 27, 30, 31}:
            return token == ""

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"</s>", "1", "<s>", "4"}:
            return k_token == "4"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"

    attn_0_1_pattern = select_closest(tokens, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {
            0,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
        }:
            return token == "3"
        elif position in {1, 20}:
            return token == "1"
        elif position in {32, 34, 35, 36, 37, 38, 39, 12, 30, 31}:
            return token == ""
        elif position in {33}:
            return token == "4"

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 4}:
            return k_position == 5
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {33, 3, 36}:
            return k_position == 4
        elif q_position in {5, 6}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 10
        elif q_position in {9, 10}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 12
        elif q_position in {27, 12}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 14
        elif q_position in {34, 14}:
            return k_position == 15
        elif q_position in {15}:
            return k_position == 16
        elif q_position in {16}:
            return k_position == 17
        elif q_position in {17}:
            return k_position == 18
        elif q_position in {18}:
            return k_position == 19
        elif q_position in {19}:
            return k_position == 20
        elif q_position in {20}:
            return k_position == 22
        elif q_position in {21}:
            return k_position == 24
        elif q_position in {22}:
            return k_position == 23
        elif q_position in {23}:
            return k_position == 26
        elif q_position in {24}:
            return k_position == 25
        elif q_position in {25}:
            return k_position == 28
        elif q_position in {26}:
            return k_position == 7
        elif q_position in {28}:
            return k_position == 0
        elif q_position in {35, 29}:
            return k_position == 27
        elif q_position in {30}:
            return k_position == 31
        elif q_position in {31}:
            return k_position == 30
        elif q_position in {32}:
            return k_position == 34
        elif q_position in {37}:
            return k_position == 33
        elif q_position in {38}:
            return k_position == 36
        elif q_position in {39}:
            return k_position == 35

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(position, token):
        if position in {0, 15, 17, 21, 22, 23, 25, 26, 27, 28, 29}:
            return token == "3"
        elif position in {24, 1, 2, 4}:
            return token == "0"
        elif position in {3, 5, 7}:
            return token == "2"
        elif position in {6, 8, 9, 10, 11, 12, 13, 14, 16, 20}:
            return token == "4"
        elif position in {32, 33, 34, 35, 36, 37, 38, 39, 18, 19, 30, 31}:
            return token == ""

    attn_0_4_pattern = select_closest(tokens, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(position, token):
        if position in {0, 11, 21, 24, 25}:
            return token == "1"
        elif position in {1, 4}:
            return token == "0"
        elif position in {2, 8, 10, 13, 14, 15, 16, 17}:
            return token == "4"
        elif position in {3, 12, 18, 19, 20, 29}:
            return token == "3"
        elif position in {5, 6, 7}:
            return token == "2"
        elif position in {32, 33, 34, 35, 36, 37, 38, 39, 9, 22, 23, 30, 31}:
            return token == ""
        elif position in {26}:
            return token == "<s>"
        elif position in {27, 28}:
            return token == "<pad>"

    attn_0_5_pattern = select_closest(tokens, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 2, 4}:
            return k_position == 1
        elif q_position in {1, 5}:
            return k_position == 3
        elif q_position in {35, 3, 30, 39}:
            return k_position == 5
        elif q_position in {36, 6, 7, 8, 10, 11, 15, 16, 20, 23, 26, 27, 28, 29}:
            return k_position == 4
        elif q_position in {9, 12, 13}:
            return k_position == 8
        elif q_position in {14}:
            return k_position == 25
        elif q_position in {32, 17}:
            return k_position == 23
        elif q_position in {18, 19}:
            return k_position == 10
        elif q_position in {21}:
            return k_position == 28
        elif q_position in {24, 22}:
            return k_position == 29
        elif q_position in {25}:
            return k_position == 2
        elif q_position in {31}:
            return k_position == 32
        elif q_position in {33}:
            return k_position == 31
        elif q_position in {34}:
            return k_position == 38
        elif q_position in {37}:
            return k_position == 16
        elif q_position in {38}:
            return k_position == 6

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0}:
            return k_position == 5
        elif q_position in {1, 9}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 7
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 9
        elif q_position in {26, 7}:
            return k_position == 8
        elif q_position in {8, 14}:
            return k_position == 1
        elif q_position in {10}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 14
        elif q_position in {12}:
            return k_position == 17
        elif q_position in {13}:
            return k_position == 20
        elif q_position in {15}:
            return k_position == 18
        elif q_position in {16}:
            return k_position == 19
        elif q_position in {17}:
            return k_position == 10
        elif q_position in {18}:
            return k_position == 21
        elif q_position in {19}:
            return k_position == 15
        elif q_position in {20}:
            return k_position == 13
        elif q_position in {21}:
            return k_position == 22
        elif q_position in {22}:
            return k_position == 25
        elif q_position in {23}:
            return k_position == 24
        elif q_position in {24, 36}:
            return k_position == 27
        elif q_position in {25}:
            return k_position == 26
        elif q_position in {27}:
            return k_position == 12
        elif q_position in {28}:
            return k_position == 0
        elif q_position in {37, 29}:
            return k_position == 16
        elif q_position in {33, 30}:
            return k_position == 29
        elif q_position in {31}:
            return k_position == 28
        elif q_position in {32}:
            return k_position == 34
        elif q_position in {34}:
            return k_position == 36
        elif q_position in {35}:
            return k_position == 35
        elif q_position in {38}:
            return k_position == 33
        elif q_position in {39}:
            return k_position == 31

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0}:
            return token == "1"
        elif position in {1, 2, 3, 4, 5}:
            return token == "0"
        elif position in {
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
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
        }:
            return token == ""
        elif position in {29}:
            return token == "</s>"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0, 8, 7}:
            return k_position == 16
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {3}:
            return k_position == 9
        elif q_position in {4}:
            return k_position == 10
        elif q_position in {32, 29, 5, 6}:
            return k_position == 13
        elif q_position in {9}:
            return k_position == 17
        elif q_position in {10}:
            return k_position == 19
        elif q_position in {11}:
            return k_position == 20
        elif q_position in {12}:
            return k_position == 24
        elif q_position in {13}:
            return k_position == 23
        elif q_position in {16, 17, 14}:
            return k_position == 26
        elif q_position in {15}:
            return k_position == 25
        elif q_position in {18}:
            return k_position == 28
        elif q_position in {19}:
            return k_position == 30
        elif q_position in {20}:
            return k_position == 38
        elif q_position in {21}:
            return k_position == 29
        elif q_position in {22}:
            return k_position == 31
        elif q_position in {27, 23}:
            return k_position == 33
        elif q_position in {24}:
            return k_position == 34
        elif q_position in {25, 26}:
            return k_position == 37
        elif q_position in {28}:
            return k_position == 36
        elif q_position in {30}:
            return k_position == 2
        elif q_position in {33, 39, 37, 31}:
            return k_position == 12
        elif q_position in {34}:
            return k_position == 8
        elif q_position in {35, 36}:
            return k_position == 11
        elif q_position in {38}:
            return k_position == 15

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0, 12, 29}:
            return k_position == 17
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {2, 3}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {5}:
            return k_position == 8
        elif q_position in {6}:
            return k_position == 9
        elif q_position in {7}:
            return k_position == 11
        elif q_position in {8}:
            return k_position == 12
        elif q_position in {9}:
            return k_position == 13
        elif q_position in {10}:
            return k_position == 14
        elif q_position in {11}:
            return k_position == 16
        elif q_position in {13}:
            return k_position == 18
        elif q_position in {14}:
            return k_position == 19
        elif q_position in {15}:
            return k_position == 20
        elif q_position in {16, 35}:
            return k_position == 21
        elif q_position in {17, 30}:
            return k_position == 22
        elif q_position in {18, 34, 38}:
            return k_position == 23
        elif q_position in {32, 19, 31}:
            return k_position == 24
        elif q_position in {20}:
            return k_position == 25
        elif q_position in {36, 21}:
            return k_position == 26
        elif q_position in {22, 39}:
            return k_position == 27
        elif q_position in {33, 23}:
            return k_position == 28
        elif q_position in {24}:
            return k_position == 29
        elif q_position in {25, 28}:
            return k_position == 33
        elif q_position in {26}:
            return k_position == 39
        elif q_position in {27}:
            return k_position == 34
        elif q_position in {37}:
            return k_position == 32

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 7
        elif q_position in {4}:
            return k_position == 8
        elif q_position in {5}:
            return k_position == 10
        elif q_position in {6}:
            return k_position == 11
        elif q_position in {8, 7}:
            return k_position == 13
        elif q_position in {9}:
            return k_position == 14
        elif q_position in {10, 29}:
            return k_position == 16
        elif q_position in {11}:
            return k_position == 18
        elif q_position in {34, 39, 12, 13, 14, 31}:
            return k_position == 20
        elif q_position in {37, 30, 15}:
            return k_position == 22
        elif q_position in {16, 32, 35}:
            return k_position == 23
        elif q_position in {17}:
            return k_position == 24
        elif q_position in {18}:
            return k_position == 25
        elif q_position in {19}:
            return k_position == 26
        elif q_position in {20, 21}:
            return k_position == 27
        elif q_position in {22}:
            return k_position == 28
        elif q_position in {23}:
            return k_position == 29
        elif q_position in {24}:
            return k_position == 33
        elif q_position in {25, 28}:
            return k_position == 32
        elif q_position in {26}:
            return k_position == 36
        elif q_position in {27}:
            return k_position == 37
        elif q_position in {33, 36, 38}:
            return k_position == 21

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {0, 27, 28}:
            return token == "1"
        elif position in {1, 2}:
            return token == "0"
        elif position in {
            3,
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
            16,
            17,
            18,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
        }:
            return token == ""
        elif position in {19, 20}:
            return token == "</s>"
        elif position in {21, 22, 23, 24, 25, 26}:
            return token == "4"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {0, 19, 20, 23, 24}:
            return token == "1"
        elif position in {1, 21, 22, 25, 26, 27, 28}:
            return token == "0"
        elif position in {
            32,
            33,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            34,
            35,
            36,
            37,
            38,
            39,
            29,
            30,
            31,
        }:
            return token == ""
        elif position in {9, 10, 11, 13, 15}:
            return token == "</s>"
        elif position in {16, 12, 14}:
            return token == "<s>"
        elif position in {17, 18}:
            return token == "4"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(q_position, k_position):
        if q_position in {0, 12}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3, 4}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 10
        elif q_position in {9}:
            return k_position == 11
        elif q_position in {10}:
            return k_position == 13
        elif q_position in {11}:
            return k_position == 14
        elif q_position in {13, 38}:
            return k_position == 16
        elif q_position in {14}:
            return k_position == 17
        elif q_position in {37, 15}:
            return k_position == 18
        elif q_position in {16}:
            return k_position == 19
        elif q_position in {17}:
            return k_position == 20
        elif q_position in {18, 30}:
            return k_position == 21
        elif q_position in {19, 20, 36}:
            return k_position == 22
        elif q_position in {33, 21}:
            return k_position == 24
        elif q_position in {31, 22, 23}:
            return k_position == 26
        elif q_position in {24, 39}:
            return k_position == 27
        elif q_position in {25, 26, 35}:
            return k_position == 28
        elif q_position in {27}:
            return k_position == 29
        elif q_position in {28}:
            return k_position == 39
        elif q_position in {32, 29}:
            return k_position == 12
        elif q_position in {34}:
            return k_position == 37

    num_attn_0_6_pattern = select(positions, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {
            0,
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
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
        }:
            return token == ""
        elif position in {1}:
            return token == "<s>"
        elif position in {2, 3, 4}:
            return token == "1"
        elif position in {29, 5, 6, 7}:
            return token == "0"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_0_output, attn_0_1_output):
        key = (attn_0_0_output, attn_0_1_output)
        return 30

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_1_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position):
        key = position
        if key in {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28}:
            return 30
        elif key in {1, 2}:
            return 7
        elif key in {3}:
            return 5
        elif key in {29}:
            return 23
        return 32

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in positions]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(position):
        key = position
        if key in {1, 2, 3}:
            return 12
        elif key in {4}:
            return 16
        elif key in {29}:
            return 24
        return 8

    mlp_0_2_outputs = [mlp_0_2(k0) for k0 in positions]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_4_output, attn_0_2_output):
        key = (attn_0_4_output, attn_0_2_output)
        return 13

    mlp_0_3_outputs = [
        mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_4_outputs, attn_0_2_outputs)
    ]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output, num_attn_0_7_output):
        key = (num_attn_0_2_output, num_attn_0_7_output)
        return 11

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_7_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output):
        key = num_attn_0_1_output
        return 35

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_1_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_3_output, num_attn_0_1_output):
        key = (num_attn_0_3_output, num_attn_0_1_output)
        if key in {(0, 0), (1, 0)}:
            return 39
        return 24

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_4_output):
        key = num_attn_0_4_output
        return 37

    num_mlp_0_3_outputs = [num_mlp_0_3(k0) for k0 in num_attn_0_4_outputs]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, num_mlp_0_2_output):
        if position in {0, 3, 9, 15, 16, 20, 21, 22}:
            return num_mlp_0_2_output == 24
        elif position in {1}:
            return num_mlp_0_2_output == 1
        elif position in {2}:
            return num_mlp_0_2_output == 39
        elif position in {4}:
            return num_mlp_0_2_output == 36
        elif position in {27, 5}:
            return num_mlp_0_2_output == 13
        elif position in {6}:
            return num_mlp_0_2_output == 0
        elif position in {32, 36, 38, 7, 10, 30}:
            return num_mlp_0_2_output == 5
        elif position in {8, 19}:
            return num_mlp_0_2_output == 30
        elif position in {11, 12}:
            return num_mlp_0_2_output == 38
        elif position in {13, 14, 31}:
            return num_mlp_0_2_output == 29
        elif position in {17}:
            return num_mlp_0_2_output == 16
        elif position in {18}:
            return num_mlp_0_2_output == 4
        elif position in {24, 25, 23}:
            return num_mlp_0_2_output == 7
        elif position in {26, 29}:
            return num_mlp_0_2_output == 2
        elif position in {28}:
            return num_mlp_0_2_output == 9
        elif position in {33, 39}:
            return num_mlp_0_2_output == 20
        elif position in {34}:
            return num_mlp_0_2_output == 25
        elif position in {35, 37}:
            return num_mlp_0_2_output == 6

    attn_1_0_pattern = select_closest(num_mlp_0_2_outputs, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_7_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_5_output, position):
        if attn_0_5_output in {"0"}:
            return position == 13
        elif attn_0_5_output in {"1"}:
            return position == 15
        elif attn_0_5_output in {"3", "2"}:
            return position == 12
        elif attn_0_5_output in {"4"}:
            return position == 6
        elif attn_0_5_output in {"</s>"}:
            return position == 3
        elif attn_0_5_output in {"<s>"}:
            return position == 8

    attn_1_1_pattern = select_closest(positions, attn_0_5_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_7_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(num_mlp_0_3_output, token):
        if num_mlp_0_3_output in {0, 33, 36, 8, 14, 15, 16, 17, 19, 20, 28, 29}:
            return token == "4"
        elif num_mlp_0_3_output in {1, 12}:
            return token == "<pad>"
        elif num_mlp_0_3_output in {
            32,
            2,
            4,
            5,
            38,
            39,
            11,
            13,
            22,
            23,
            24,
            25,
            30,
            31,
        }:
            return token == ""
        elif num_mlp_0_3_output in {3, 35, 37, 6, 7, 9, 21}:
            return token == "3"
        elif num_mlp_0_3_output in {10, 26, 34}:
            return token == "2"
        elif num_mlp_0_3_output in {18, 27}:
            return token == "0"

    attn_1_2_pattern = select_closest(tokens, num_mlp_0_3_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0, 2}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {17, 3}:
            return k_position == 4
        elif q_position in {8, 4}:
            return k_position == 2
        elif q_position in {35, 36, 5, 38, 12, 13}:
            return k_position == 7
        elif q_position in {16, 34, 6}:
            return k_position == 15
        elif q_position in {32, 31, 7}:
            return k_position == 5
        elif q_position in {9}:
            return k_position == 6
        elif q_position in {10, 15}:
            return k_position == 14
        elif q_position in {11}:
            return k_position == 13
        elif q_position in {14}:
            return k_position == 16
        elif q_position in {18, 26}:
            return k_position == 12
        elif q_position in {19}:
            return k_position == 23
        elif q_position in {20}:
            return k_position == 17
        elif q_position in {21}:
            return k_position == 28
        elif q_position in {28, 29, 22, 30}:
            return k_position == 27
        elif q_position in {39, 23}:
            return k_position == 11
        elif q_position in {24}:
            return k_position == 25
        elif q_position in {25}:
            return k_position == 21
        elif q_position in {27}:
            return k_position == 19
        elif q_position in {33}:
            return k_position == 0
        elif q_position in {37}:
            return k_position == 18

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(attn_0_1_output, attn_0_5_output):
        if attn_0_1_output in {"2", "0"}:
            return attn_0_5_output == "1"
        elif attn_0_1_output in {"1", "<s>", "4"}:
            return attn_0_5_output == ""
        elif attn_0_1_output in {"3"}:
            return attn_0_5_output == "</s>"
        elif attn_0_1_output in {"</s>"}:
            return attn_0_5_output == "<s>"

    attn_1_4_pattern = select_closest(attn_0_5_outputs, attn_0_1_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_2_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(position, token):
        if position in {0, 2, 19, 20, 21, 26}:
            return token == "1"
        elif position in {1, 34, 22, 27, 31}:
            return token == "0"
        elif position in {3, 6, 7, 8, 9, 39, 11, 12, 13, 14, 15, 16, 17, 24, 30}:
            return token == "4"
        elif position in {4, 5, 10, 18, 25, 29}:
            return token == "3"
        elif position in {23}:
            return token == "2"
        elif position in {28}:
            return token == "<s>"
        elif position in {32, 33, 35, 36, 38}:
            return token == ""
        elif position in {37}:
            return token == "</s>"

    attn_1_5_pattern = select_closest(tokens, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, tokens)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(mlp_0_3_output, mlp_0_2_output):
        if mlp_0_3_output in {0, 30}:
            return mlp_0_2_output == 37
        elif mlp_0_3_output in {1, 11}:
            return mlp_0_2_output == 30
        elif mlp_0_3_output in {2}:
            return mlp_0_2_output == 39
        elif mlp_0_3_output in {10, 3}:
            return mlp_0_2_output == 35
        elif mlp_0_3_output in {4}:
            return mlp_0_2_output == 19
        elif mlp_0_3_output in {5, 14}:
            return mlp_0_2_output == 5
        elif mlp_0_3_output in {38, 29, 6}:
            return mlp_0_2_output == 3
        elif mlp_0_3_output in {7}:
            return mlp_0_2_output == 0
        elif mlp_0_3_output in {8, 35, 13, 23}:
            return mlp_0_2_output == 8
        elif mlp_0_3_output in {33, 34, 36, 39, 9, 17, 31}:
            return mlp_0_2_output == 4
        elif mlp_0_3_output in {32, 12}:
            return mlp_0_2_output == 1
        elif mlp_0_3_output in {24, 15}:
            return mlp_0_2_output == 18
        elif mlp_0_3_output in {16, 18}:
            return mlp_0_2_output == 12
        elif mlp_0_3_output in {19}:
            return mlp_0_2_output == 25
        elif mlp_0_3_output in {25, 20}:
            return mlp_0_2_output == 33
        elif mlp_0_3_output in {21}:
            return mlp_0_2_output == 6
        elif mlp_0_3_output in {26, 27, 28, 22}:
            return mlp_0_2_output == 7
        elif mlp_0_3_output in {37}:
            return mlp_0_2_output == 26

    attn_1_6_pattern = select_closest(mlp_0_2_outputs, mlp_0_3_outputs, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_1_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(q_position, k_position):
        if q_position in {0, 28, 29}:
            return k_position == 1
        elif q_position in {32, 1, 5}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {33, 4, 7, 8, 14}:
            return k_position == 6
        elif q_position in {34, 38, 6, 9, 19}:
            return k_position == 7
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {11, 13}:
            return k_position == 10
        elif q_position in {25, 35, 12}:
            return k_position == 15
        elif q_position in {16, 15}:
            return k_position == 13
        elif q_position in {17, 26, 36, 31}:
            return k_position == 8
        elif q_position in {18}:
            return k_position == 11
        elif q_position in {20}:
            return k_position == 16
        elif q_position in {21, 23}:
            return k_position == 28
        elif q_position in {22}:
            return k_position == 24
        elif q_position in {24, 37, 39}:
            return k_position == 14
        elif q_position in {27}:
            return k_position == 20
        elif q_position in {30}:
            return k_position == 19

    attn_1_7_pattern = select_closest(positions, positions, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_1_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, attn_0_7_output):
        if position in {
            0,
            10,
            11,
            12,
            13,
            14,
            15,
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
        }:
            return attn_0_7_output == ""
        elif position in {1, 29, 5}:
            return attn_0_7_output == "1"
        elif position in {32, 33, 2, 3, 4, 34, 6, 7, 8, 35, 36, 37, 38, 39, 31}:
            return attn_0_7_output == "0"
        elif position in {9}:
            return attn_0_7_output == "</s>"
        elif position in {16}:
            return attn_0_7_output == "<pad>"

    num_attn_1_0_pattern = select(attn_0_7_outputs, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_0_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(num_mlp_0_0_output, token):
        if num_mlp_0_0_output in {
            0,
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
            15,
            16,
            17,
            18,
            19,
            22,
            24,
            25,
            27,
            28,
            30,
            31,
            32,
            33,
            34,
            35,
            38,
            39,
        }:
            return token == "2"
        elif num_mlp_0_0_output in {1, 26, 3, 14}:
            return token == "1"
        elif num_mlp_0_0_output in {20, 36}:
            return token == "<s>"
        elif num_mlp_0_0_output in {21}:
            return token == ""
        elif num_mlp_0_0_output in {23}:
            return token == "3"
        elif num_mlp_0_0_output in {29}:
            return token == "<pad>"
        elif num_mlp_0_0_output in {37}:
            return token == "0"

    num_attn_1_1_pattern = select(tokens, num_mlp_0_0_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, ones)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, token):
        if position in {0, 32, 2, 34, 8, 9, 10}:
            return token == "0"
        elif position in {1}:
            return token == "</s>"
        elif position in {33, 3, 4, 5, 6, 7, 35, 36, 37, 38, 39}:
            return token == "1"
        elif position in {
            11,
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
            29,
            30,
            31,
        }:
            return token == ""
        elif position in {12}:
            return token == "<s>"

    num_attn_1_2_pattern = select(tokens, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, ones)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, attn_0_7_output):
        if position in {
            0,
            5,
            7,
            9,
            12,
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
            30,
            31,
            33,
            34,
            35,
            36,
            38,
            39,
        }:
            return attn_0_7_output == ""
        elif position in {1, 11, 13}:
            return attn_0_7_output == "1"
        elif position in {8, 2, 10, 6}:
            return attn_0_7_output == "2"
        elif position in {32, 3, 4, 37, 15}:
            return attn_0_7_output == "0"
        elif position in {17, 14}:
            return attn_0_7_output == "<s>"
        elif position in {16}:
            return attn_0_7_output == "</s>"

    num_attn_1_3_pattern = select(attn_0_7_outputs, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(mlp_0_3_output, attn_0_7_output):
        if mlp_0_3_output in {0, 4, 6, 7, 8, 12}:
            return attn_0_7_output == "2"
        elif mlp_0_3_output in {
            1,
            9,
            10,
            11,
            13,
            14,
            16,
            17,
            18,
            19,
            21,
            22,
            25,
            26,
            27,
            30,
            31,
            33,
            35,
            38,
        }:
            return attn_0_7_output == "0"
        elif mlp_0_3_output in {32, 2, 3, 34, 5, 36, 37, 39, 15, 20, 23, 24, 28, 29}:
            return attn_0_7_output == "1"

    num_attn_1_4_pattern = select(attn_0_7_outputs, mlp_0_3_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_7_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(position, attn_0_5_output):
        if position in {0, 34, 37, 23, 24, 25, 26, 28, 30}:
            return attn_0_5_output == ""
        elif position in {1, 5, 8, 12, 13, 14, 20, 27, 31}:
            return attn_0_5_output == "1"
        elif position in {
            32,
            33,
            2,
            4,
            36,
            6,
            38,
            39,
            9,
            10,
            15,
            16,
            17,
            18,
            19,
            21,
            29,
        }:
            return attn_0_5_output == "0"
        elif position in {11, 35, 3, 7}:
            return attn_0_5_output == "2"
        elif position in {22}:
            return attn_0_5_output == "3"

    num_attn_1_5_pattern = select(attn_0_5_outputs, positions, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_0_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(position, token):
        if position in {0, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28}:
            return token == ""
        elif position in {1, 4}:
            return token == "2"
        elif position in {32, 33, 2, 34, 35, 5, 6, 7, 8, 9, 10, 36, 37, 38, 39, 30, 31}:
            return token == "1"
        elif position in {3, 11, 12, 13, 14, 29}:
            return token == "0"
        elif position in {15}:
            return token == "</s>"

    num_attn_1_6_pattern = select(tokens, positions, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, ones)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(mlp_0_0_output, token):
        if mlp_0_0_output in {
            0,
            1,
            2,
            3,
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
            33,
            34,
            35,
            36,
            37,
            38,
        }:
            return token == ""
        elif mlp_0_0_output in {29}:
            return token == "2"
        elif mlp_0_0_output in {39, 31}:
            return token == "<pad>"
        elif mlp_0_0_output in {32}:
            return token == "</s>"

    num_attn_1_7_pattern = select(tokens, mlp_0_0_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_4_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_1_output, attn_1_6_output):
        key = (attn_0_1_output, attn_1_6_output)
        if key in {
            ("0", "0"),
            ("0", "<s>"),
            ("2", "0"),
            ("2", "<s>"),
            ("3", "0"),
            ("3", "2"),
            ("3", "<s>"),
            ("4", "0"),
            ("4", "<s>"),
            ("</s>", "0"),
            ("</s>", "<s>"),
            ("<s>", "0"),
            ("<s>", "2"),
            ("<s>", "<s>"),
        }:
            return 26
        return 3

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_1_6_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(mlp_0_1_output, attn_1_4_output):
        key = (mlp_0_1_output, attn_1_4_output)
        if key in {
            (9, "0"),
            (9, "1"),
            (9, "4"),
            (15, "0"),
            (15, "1"),
            (15, "4"),
            (16, "0"),
            (16, "1"),
            (16, "4"),
            (20, "1"),
            (25, "0"),
            (25, "1"),
        }:
            return 33
        return 19

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(mlp_0_1_outputs, attn_1_4_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(mlp_0_3_output, attn_0_7_output):
        key = (mlp_0_3_output, attn_0_7_output)
        return 28

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(mlp_0_3_outputs, attn_0_7_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(position, attn_1_0_output):
        key = (position, attn_1_0_output)
        return 13

    mlp_1_3_outputs = [mlp_1_3(k0, k1) for k0, k1 in zip(positions, attn_1_0_outputs)]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_1_output):
        key = num_attn_0_1_output
        if key in {0}:
            return 33
        return 13

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_0_1_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_7_output, num_attn_0_0_output):
        key = (num_attn_1_7_output, num_attn_0_0_output)
        return 19

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_0_3_output, num_attn_0_2_output):
        key = (num_attn_0_3_output, num_attn_0_2_output)
        if key in {(0, 0), (0, 1)}:
            return 31
        return 21

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_7_output, num_attn_1_1_output):
        key = (num_attn_1_7_output, num_attn_1_1_output)
        return 18

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_position, k_position):
        if q_position in {0, 35, 30}:
            return k_position == 7
        elif q_position in {1, 34, 29}:
            return k_position == 2
        elif q_position in {2, 3, 8, 15, 16}:
            return k_position == 4
        elif q_position in {11, 4, 5, 7}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {32, 33, 36, 37, 9, 31}:
            return k_position == 8
        elif q_position in {10, 14}:
            return k_position == 13
        elif q_position in {12, 13}:
            return k_position == 10
        elif q_position in {17, 23}:
            return k_position == 28
        elif q_position in {18, 26}:
            return k_position == 14
        elif q_position in {19, 20}:
            return k_position == 17
        elif q_position in {24, 21}:
            return k_position == 25
        elif q_position in {25, 22}:
            return k_position == 29
        elif q_position in {27}:
            return k_position == 18
        elif q_position in {28, 38}:
            return k_position == 21
        elif q_position in {39}:
            return k_position == 36

    attn_2_0_pattern = select_closest(positions, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_5_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_num_mlp_1_0_output, k_num_mlp_1_0_output):
        if q_num_mlp_1_0_output in {0}:
            return k_num_mlp_1_0_output == 22
        elif q_num_mlp_1_0_output in {1, 34, 39, 13, 18, 28}:
            return k_num_mlp_1_0_output == 33
        elif q_num_mlp_1_0_output in {2}:
            return k_num_mlp_1_0_output == 3
        elif q_num_mlp_1_0_output in {3, 30}:
            return k_num_mlp_1_0_output == 13
        elif q_num_mlp_1_0_output in {17, 4}:
            return k_num_mlp_1_0_output == 20
        elif q_num_mlp_1_0_output in {5}:
            return k_num_mlp_1_0_output == 8
        elif q_num_mlp_1_0_output in {27, 36, 6}:
            return k_num_mlp_1_0_output == 15
        elif q_num_mlp_1_0_output in {11, 7}:
            return k_num_mlp_1_0_output == 12
        elif q_num_mlp_1_0_output in {8, 37}:
            return k_num_mlp_1_0_output == 0
        elif q_num_mlp_1_0_output in {32, 9, 35, 20}:
            return k_num_mlp_1_0_output == 2
        elif q_num_mlp_1_0_output in {10}:
            return k_num_mlp_1_0_output == 38
        elif q_num_mlp_1_0_output in {12}:
            return k_num_mlp_1_0_output == 1
        elif q_num_mlp_1_0_output in {14}:
            return k_num_mlp_1_0_output == 35
        elif q_num_mlp_1_0_output in {15}:
            return k_num_mlp_1_0_output == 39
        elif q_num_mlp_1_0_output in {16}:
            return k_num_mlp_1_0_output == 9
        elif q_num_mlp_1_0_output in {19}:
            return k_num_mlp_1_0_output == 24
        elif q_num_mlp_1_0_output in {21}:
            return k_num_mlp_1_0_output == 27
        elif q_num_mlp_1_0_output in {38, 22}:
            return k_num_mlp_1_0_output == 6
        elif q_num_mlp_1_0_output in {23}:
            return k_num_mlp_1_0_output == 29
        elif q_num_mlp_1_0_output in {24}:
            return k_num_mlp_1_0_output == 34
        elif q_num_mlp_1_0_output in {25, 31}:
            return k_num_mlp_1_0_output == 36
        elif q_num_mlp_1_0_output in {26}:
            return k_num_mlp_1_0_output == 31
        elif q_num_mlp_1_0_output in {29}:
            return k_num_mlp_1_0_output == 26
        elif q_num_mlp_1_0_output in {33}:
            return k_num_mlp_1_0_output == 7

    attn_2_1_pattern = select_closest(
        num_mlp_1_0_outputs, num_mlp_1_0_outputs, predicate_2_1
    )
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_7_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(mlp_0_2_output, num_mlp_0_3_output):
        if mlp_0_2_output in {0, 33, 5}:
            return num_mlp_0_3_output == 7
        elif mlp_0_2_output in {1}:
            return num_mlp_0_3_output == 31
        elif mlp_0_2_output in {2}:
            return num_mlp_0_3_output == 19
        elif mlp_0_2_output in {9, 3, 20}:
            return num_mlp_0_3_output == 39
        elif mlp_0_2_output in {4}:
            return num_mlp_0_3_output == 16
        elif mlp_0_2_output in {23, 6, 22}:
            return num_mlp_0_3_output == 28
        elif mlp_0_2_output in {31, 7}:
            return num_mlp_0_3_output == 38
        elif mlp_0_2_output in {8, 26}:
            return num_mlp_0_3_output == 6
        elif mlp_0_2_output in {10, 13}:
            return num_mlp_0_3_output == 11
        elif mlp_0_2_output in {11, 12}:
            return num_mlp_0_3_output == 1
        elif mlp_0_2_output in {28, 14}:
            return num_mlp_0_3_output == 34
        elif mlp_0_2_output in {15}:
            return num_mlp_0_3_output == 5
        elif mlp_0_2_output in {16, 25}:
            return num_mlp_0_3_output == 32
        elif mlp_0_2_output in {34, 35, 37, 38, 39, 17, 27}:
            return num_mlp_0_3_output == 3
        elif mlp_0_2_output in {18, 19}:
            return num_mlp_0_3_output == 18
        elif mlp_0_2_output in {32, 21}:
            return num_mlp_0_3_output == 0
        elif mlp_0_2_output in {24}:
            return num_mlp_0_3_output == 9
        elif mlp_0_2_output in {29}:
            return num_mlp_0_3_output == 14
        elif mlp_0_2_output in {30}:
            return num_mlp_0_3_output == 4
        elif mlp_0_2_output in {36}:
            return num_mlp_0_3_output == 27

    attn_2_2_pattern = select_closest(
        num_mlp_0_3_outputs, mlp_0_2_outputs, predicate_2_2
    )
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_num_mlp_1_2_output, k_num_mlp_1_2_output):
        if q_num_mlp_1_2_output in {0}:
            return k_num_mlp_1_2_output == 16
        elif q_num_mlp_1_2_output in {1, 9}:
            return k_num_mlp_1_2_output == 15
        elif q_num_mlp_1_2_output in {16, 2, 28}:
            return k_num_mlp_1_2_output == 1
        elif q_num_mlp_1_2_output in {27, 3, 5}:
            return k_num_mlp_1_2_output == 24
        elif q_num_mlp_1_2_output in {4, 6}:
            return k_num_mlp_1_2_output == 19
        elif q_num_mlp_1_2_output in {7, 14, 19, 21, 23, 24, 26}:
            return k_num_mlp_1_2_output == 31
        elif q_num_mlp_1_2_output in {8, 33, 10}:
            return k_num_mlp_1_2_output == 9
        elif q_num_mlp_1_2_output in {11}:
            return k_num_mlp_1_2_output == 22
        elif q_num_mlp_1_2_output in {31, 12, 20, 15}:
            return k_num_mlp_1_2_output == 21
        elif q_num_mlp_1_2_output in {13}:
            return k_num_mlp_1_2_output == 4
        elif q_num_mlp_1_2_output in {17, 29}:
            return k_num_mlp_1_2_output == 8
        elif q_num_mlp_1_2_output in {18}:
            return k_num_mlp_1_2_output == 28
        elif q_num_mlp_1_2_output in {35, 39, 22, 30}:
            return k_num_mlp_1_2_output == 14
        elif q_num_mlp_1_2_output in {25, 37}:
            return k_num_mlp_1_2_output == 33
        elif q_num_mlp_1_2_output in {32}:
            return k_num_mlp_1_2_output == 6
        elif q_num_mlp_1_2_output in {34}:
            return k_num_mlp_1_2_output == 25
        elif q_num_mlp_1_2_output in {36}:
            return k_num_mlp_1_2_output == 38
        elif q_num_mlp_1_2_output in {38}:
            return k_num_mlp_1_2_output == 7

    attn_2_3_pattern = select_closest(
        num_mlp_1_2_outputs, num_mlp_1_2_outputs, predicate_2_3
    )
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_0_1_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "</s>"
        elif q_token in {"1", "2"}:
            return k_token == "3"
        elif q_token in {"3"}:
            return k_token == "2"
        elif q_token in {"4"}:
            return k_token == "0"
        elif q_token in {"</s>", "<s>"}:
            return k_token == "<s>"

    attn_2_4_pattern = select_closest(tokens, tokens, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_2_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(token, num_mlp_1_3_output):
        if token in {"0"}:
            return num_mlp_1_3_output == 4
        elif token in {"1", "3"}:
            return num_mlp_1_3_output == 31
        elif token in {"2"}:
            return num_mlp_1_3_output == 2
        elif token in {"4"}:
            return num_mlp_1_3_output == 26
        elif token in {"</s>", "<s>"}:
            return num_mlp_1_3_output == 7

    attn_2_5_pattern = select_closest(num_mlp_1_3_outputs, tokens, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_0_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(token, mlp_1_3_output):
        if token in {"0"}:
            return mlp_1_3_output == 29
        elif token in {"1", "2", "4"}:
            return mlp_1_3_output == 6
        elif token in {"</s>", "3"}:
            return mlp_1_3_output == 7
        elif token in {"<s>"}:
            return mlp_1_3_output == 28

    attn_2_6_pattern = select_closest(mlp_1_3_outputs, tokens, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_3_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(mlp_0_2_output, attn_1_0_output):
        if mlp_0_2_output in {
            0,
            1,
            32,
            37,
            6,
            7,
            39,
            9,
            11,
            18,
            19,
            21,
            23,
            24,
            25,
            28,
        }:
            return attn_1_0_output == ""
        elif mlp_0_2_output in {2, 14, 17, 22, 30}:
            return attn_1_0_output == "4"
        elif mlp_0_2_output in {27, 35, 10, 3}:
            return attn_1_0_output == "<pad>"
        elif mlp_0_2_output in {34, 4, 20}:
            return attn_1_0_output == "2"
        elif mlp_0_2_output in {8, 16, 12, 5}:
            return attn_1_0_output == "<s>"
        elif mlp_0_2_output in {33, 13}:
            return attn_1_0_output == "1"
        elif mlp_0_2_output in {29, 38, 15}:
            return attn_1_0_output == "0"
        elif mlp_0_2_output in {26}:
            return attn_1_0_output == "</s>"
        elif mlp_0_2_output in {36, 31}:
            return attn_1_0_output == "3"

    attn_2_7_pattern = select_closest(attn_1_0_outputs, mlp_0_2_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_2_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_1_output, attn_1_3_output):
        if attn_1_1_output in {"</s>", "1", "3", "<s>", "4", "0"}:
            return attn_1_3_output == ""
        elif attn_1_1_output in {"2"}:
            return attn_1_3_output == "<s>"

    num_attn_2_0_pattern = select(attn_1_3_outputs, attn_1_1_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_7_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, attn_1_3_output):
        if position in {0, 33, 34, 36, 37, 38, 39, 8, 11, 12, 29}:
            return attn_1_3_output == "1"
        elif position in {1, 3, 13, 15}:
            return attn_1_3_output == "<s>"
        elif position in {32, 2, 35, 5, 6, 7, 9, 10}:
            return attn_1_3_output == "2"
        elif position in {4}:
            return attn_1_3_output == "</s>"
        elif position in {
            14,
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
            31,
        }:
            return attn_1_3_output == ""

    num_attn_2_1_pattern = select(attn_1_3_outputs, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_7_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(position, token):
        if position in {0, 35, 36, 31}:
            return token == "1"
        elif position in {1, 10, 11}:
            return token == "<s>"
        elif position in {32, 2, 3, 4, 37, 6, 7, 8, 9, 38, 39}:
            return token == "0"
        elif position in {5}:
            return token == "2"
        elif position in {
            12,
            13,
            14,
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
            29,
            30,
            33,
            34,
        }:
            return token == ""
        elif position in {15}:
            return token == "<pad>"

    num_attn_2_2_pattern = select(tokens, positions, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_5_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_4_output, position):
        if attn_1_4_output in {"0"}:
            return position == 21
        elif attn_1_4_output in {"1"}:
            return position == 4
        elif attn_1_4_output in {"</s>", "2"}:
            return position == 12
        elif attn_1_4_output in {"3"}:
            return position == 3
        elif attn_1_4_output in {"4"}:
            return position == 10
        elif attn_1_4_output in {"<s>"}:
            return position == 2

    num_attn_2_3_pattern = select(positions, attn_1_4_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_7_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_1_4_output, attn_1_5_output):
        if attn_1_4_output in {"</s>", "2", "0"}:
            return attn_1_5_output == "1"
        elif attn_1_4_output in {"1"}:
            return attn_1_5_output == "0"
        elif attn_1_4_output in {"3"}:
            return attn_1_5_output == ""
        elif attn_1_4_output in {"<s>", "4"}:
            return attn_1_5_output == "2"

    num_attn_2_4_pattern = select(attn_1_5_outputs, attn_1_4_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_5_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_1_4_output, attn_0_0_output):
        if attn_1_4_output in {"</s>", "0"}:
            return attn_0_0_output == "1"
        elif attn_1_4_output in {"1", "3", "4"}:
            return attn_0_0_output == ""
        elif attn_1_4_output in {"<s>", "2"}:
            return attn_0_0_output == "0"

    num_attn_2_5_pattern = select(attn_0_0_outputs, attn_1_4_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_0_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_4_output, attn_1_5_output):
        if attn_1_4_output in {"<s>", "2", "0"}:
            return attn_1_5_output == ""
        elif attn_1_4_output in {"</s>", "1"}:
            return attn_1_5_output == "</s>"
        elif attn_1_4_output in {"3"}:
            return attn_1_5_output == "0"
        elif attn_1_4_output in {"4"}:
            return attn_1_5_output == "1"

    num_attn_2_6_pattern = select(attn_1_5_outputs, attn_1_4_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_0_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(position, attn_0_4_output):
        if position in {
            0,
            7,
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
            28,
            29,
            30,
            31,
            33,
        }:
            return attn_0_4_output == ""
        elif position in {1}:
            return attn_0_4_output == "<s>"
        elif position in {2, 35, 3, 4, 34, 36, 37, 38, 39}:
            return attn_0_4_output == "0"
        elif position in {5, 6}:
            return attn_0_4_output == "</s>"
        elif position in {32, 27}:
            return attn_0_4_output == "<pad>"

    num_attn_2_7_pattern = select(attn_0_4_outputs, positions, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_0_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_0_6_output, num_mlp_0_2_output):
        key = (attn_0_6_output, num_mlp_0_2_output)
        if key in {
            ("1", 0),
            ("1", 9),
            ("1", 10),
            ("1", 11),
            ("1", 13),
            ("1", 18),
            ("1", 19),
            ("1", 21),
            ("1", 22),
            ("1", 23),
            ("1", 24),
            ("1", 32),
            ("4", 10),
            ("4", 11),
            ("4", 13),
            ("4", 19),
            ("4", 22),
            ("4", 23),
            ("4", 24),
            ("4", 32),
            ("</s>", 10),
            ("</s>", 11),
            ("</s>", 13),
            ("</s>", 19),
            ("</s>", 22),
            ("</s>", 23),
            ("</s>", 24),
            ("</s>", 32),
            ("<s>", 0),
            ("<s>", 8),
            ("<s>", 9),
            ("<s>", 10),
            ("<s>", 11),
            ("<s>", 13),
            ("<s>", 17),
            ("<s>", 18),
            ("<s>", 19),
            ("<s>", 21),
            ("<s>", 22),
            ("<s>", 23),
            ("<s>", 24),
            ("<s>", 26),
            ("<s>", 27),
            ("<s>", 32),
            ("<s>", 34),
            ("<s>", 36),
        }:
            return 4
        elif key in {
            ("1", 26),
            ("1", 34),
            ("4", 9),
            ("4", 21),
            ("<s>", 20),
            ("<s>", 28),
            ("<s>", 31),
            ("<s>", 37),
        }:
            return 27
        elif key in {("1", 17), ("4", 18), ("<s>", 12)}:
            return 39
        return 16

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_0_6_outputs, num_mlp_0_2_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_0_6_output, attn_1_5_output):
        key = (attn_0_6_output, attn_1_5_output)
        return 7

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_1_5_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(position, attn_2_6_output):
        key = (position, attn_2_6_output)
        if key in {
            (0, "0"),
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "</s>"),
            (1, "<s>"),
            (2, "0"),
            (3, "0"),
            (3, "1"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (3, "</s>"),
            (3, "<s>"),
            (16, "0"),
            (17, "0"),
            (18, "0"),
            (24, "0"),
            (29, "0"),
            (30, "0"),
            (31, "0"),
            (32, "0"),
            (33, "0"),
            (34, "0"),
            (35, "0"),
            (36, "0"),
            (37, "0"),
            (38, "0"),
            (39, "0"),
        }:
            return 29
        return 2

    mlp_2_2_outputs = [mlp_2_2(k0, k1) for k0, k1 in zip(positions, attn_2_6_outputs)]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(attn_1_6_output, attn_1_2_output):
        key = (attn_1_6_output, attn_1_2_output)
        return 4

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(attn_1_6_outputs, attn_1_2_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_0_2_output, num_attn_0_6_output):
        key = (num_attn_0_2_output, num_attn_0_6_output)
        if key in {(0, 0)}:
            return 24
        return 37

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_5_output, num_attn_1_1_output):
        key = (num_attn_2_5_output, num_attn_1_1_output)
        return 26

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_5_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_0_3_output):
        key = num_attn_0_3_output
        if key in {0}:
            return 34
        return 23

    num_mlp_2_2_outputs = [num_mlp_2_2(k0) for k0 in num_attn_0_3_outputs]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_2_7_output, num_attn_1_7_output):
        key = (num_attn_2_7_output, num_attn_1_7_output)
        if key in {(0, 0)}:
            return 32
        return 21

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_2_7_outputs, num_attn_1_7_outputs)
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


print(run(["<s>", "0", "3", "2", "3", "0", "2", "1", "3", "2", "</s>"]))
