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
        "output/length/rasp/sort/trainlength40/s3/sort_weights.csv",
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
    def predicate_0_0(q_position, k_position):
        if q_position in {0, 3}:
            return k_position == 5
        elif q_position in {1, 31}:
            return k_position == 3
        elif q_position in {2, 34}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 10
        elif q_position in {42, 5, 23}:
            return k_position == 9
        elif q_position in {6, 14}:
            return k_position == 8
        elif q_position in {8, 7}:
            return k_position == 13
        elif q_position in {9, 12}:
            return k_position == 15
        elif q_position in {10}:
            return k_position == 25
        elif q_position in {11}:
            return k_position == 14
        elif q_position in {13}:
            return k_position == 16
        elif q_position in {49, 18, 15}:
            return k_position == 22
        elif q_position in {16}:
            return k_position == 20
        elif q_position in {17, 20}:
            return k_position == 24
        elif q_position in {19}:
            return k_position == 26
        elif q_position in {26, 21}:
            return k_position == 11
        elif q_position in {25, 22}:
            return k_position == 29
        elif q_position in {24}:
            return k_position == 31
        elif q_position in {32, 27, 47}:
            return k_position == 36
        elif q_position in {39, 48, 28, 29, 30}:
            return k_position == 39
        elif q_position in {33}:
            return k_position == 37
        elif q_position in {35, 36, 37, 38}:
            return k_position == 0
        elif q_position in {40}:
            return k_position == 28
        elif q_position in {41}:
            return k_position == 48
        elif q_position in {43}:
            return k_position == 45
        elif q_position in {44}:
            return k_position == 32
        elif q_position in {45, 46}:
            return k_position == 27

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 29
        elif q_position in {1, 2, 41, 25, 27}:
            return k_position == 3
        elif q_position in {35, 34, 3, 6}:
            return k_position == 4
        elif q_position in {10, 4, 36}:
            return k_position == 6
        elif q_position in {5, 7}:
            return k_position == 2
        elif q_position in {8, 16}:
            return k_position == 7
        elif q_position in {9, 18, 39}:
            return k_position == 17
        elif q_position in {11}:
            return k_position == 13
        elif q_position in {12, 46}:
            return k_position == 16
        elif q_position in {13}:
            return k_position == 21
        elif q_position in {14}:
            return k_position == 10
        elif q_position in {26, 45, 15}:
            return k_position == 12
        elif q_position in {17}:
            return k_position == 11
        elif q_position in {33, 19}:
            return k_position == 9
        elif q_position in {20, 44}:
            return k_position == 25
        elif q_position in {28, 21}:
            return k_position == 35
        elif q_position in {22}:
            return k_position == 36
        elif q_position in {43, 23}:
            return k_position == 8
        elif q_position in {24}:
            return k_position == 19
        elif q_position in {29}:
            return k_position == 37
        elif q_position in {30}:
            return k_position == 20
        elif q_position in {32, 48, 31}:
            return k_position == 38
        elif q_position in {37}:
            return k_position == 24
        elif q_position in {38}:
            return k_position == 23
        elif q_position in {40}:
            return k_position == 46
        elif q_position in {42}:
            return k_position == 33
        elif q_position in {47}:
            return k_position == 41
        elif q_position in {49}:
            return k_position == 44

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_token, k_token):
        if q_token in {"<s>", "0", "3", "1", "</s>"}:
            return k_token == "3"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"4"}:
            return k_token == "4"

    attn_0_2_pattern = select_closest(tokens, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 1
        elif q_position in {1, 9}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 8
        elif q_position in {11, 4, 39}:
            return k_position == 9
        elif q_position in {12, 5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 12
        elif q_position in {24, 7}:
            return k_position == 5
        elif q_position in {8, 40}:
            return k_position == 11
        elif q_position in {10}:
            return k_position == 7
        elif q_position in {13}:
            return k_position == 17
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 20
        elif q_position in {16, 41}:
            return k_position == 21
        elif q_position in {17, 49}:
            return k_position == 19
        elif q_position in {18, 42, 21, 46}:
            return k_position == 23
        elif q_position in {19}:
            return k_position == 24
        elif q_position in {20}:
            return k_position == 14
        elif q_position in {45, 22}:
            return k_position == 10
        elif q_position in {23}:
            return k_position == 28
        elif q_position in {25}:
            return k_position == 15
        elif q_position in {32, 26}:
            return k_position == 35
        elif q_position in {27, 31}:
            return k_position == 37
        elif q_position in {28}:
            return k_position == 34
        elif q_position in {29, 30}:
            return k_position == 36
        elif q_position in {33}:
            return k_position == 38
        elif q_position in {34}:
            return k_position == 22
        elif q_position in {35, 36, 37}:
            return k_position == 0
        elif q_position in {38, 47}:
            return k_position == 39
        elif q_position in {43}:
            return k_position == 40
        elif q_position in {44}:
            return k_position == 16
        elif q_position in {48}:
            return k_position == 44

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3, 14}:
            return k_position == 15
        elif q_position in {4, 45, 15}:
            return k_position == 16
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8, 42}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 12
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 17
        elif q_position in {17}:
            return k_position == 18
        elif q_position in {18}:
            return k_position == 19
        elif q_position in {19}:
            return k_position == 20
        elif q_position in {20}:
            return k_position == 21
        elif q_position in {21}:
            return k_position == 22
        elif q_position in {41, 22}:
            return k_position == 23
        elif q_position in {46, 23}:
            return k_position == 25
        elif q_position in {24}:
            return k_position == 26
        elif q_position in {25}:
            return k_position == 27
        elif q_position in {40, 49, 26}:
            return k_position == 28
        elif q_position in {27}:
            return k_position == 29
        elif q_position in {28}:
            return k_position == 30
        elif q_position in {29, 30}:
            return k_position == 31
        elif q_position in {31}:
            return k_position == 32
        elif q_position in {32}:
            return k_position == 34
        elif q_position in {48, 33}:
            return k_position == 35
        elif q_position in {34, 36}:
            return k_position == 0
        elif q_position in {35}:
            return k_position == 37
        elif q_position in {37}:
            return k_position == 38
        elif q_position in {38}:
            return k_position == 39
        elif q_position in {39}:
            return k_position == 3
        elif q_position in {43}:
            return k_position == 42
        elif q_position in {44}:
            return k_position == 46
        elif q_position in {47}:
            return k_position == 47

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(position, token):
        if position in {0, 36, 11, 23, 25}:
            return token == "1"
        elif position in {
            1,
            10,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            24,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            37,
            38,
        }:
            return token == "3"
        elif position in {2, 3, 4, 5, 6, 7, 8, 9, 39, 12, 13}:
            return token == "2"
        elif position in {22}:
            return token == "0"
        elif position in {40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 26}:
            return token == ""

    attn_0_5_pattern = select_closest(tokens, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(position, token):
        if position in {0, 24}:
            return token == "1"
        elif position in {1, 2, 3, 9, 15, 18, 20, 21}:
            return token == "2"
        elif position in {32, 33, 34, 35, 4, 36, 6, 37, 38, 39, 13, 14, 23, 27}:
            return token == "3"
        elif position in {5, 10, 12, 16, 17, 19}:
            return token == "4"
        elif position in {7, 8, 40, 42, 11, 43, 44, 45, 46, 48, 49, 26}:
            return token == ""
        elif position in {47, 22, 28, 29, 30}:
            return token == "0"
        elif position in {25}:
            return token == "<pad>"
        elif position in {41, 31}:
            return token == "</s>"

    attn_0_6_pattern = select_closest(tokens, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(position, token):
        if position in {0, 34, 3, 4, 5, 36, 38, 39, 9, 10, 13, 18, 24, 28}:
            return token == "3"
        elif position in {1, 11, 6}:
            return token == "2"
        elif position in {32, 33, 2, 8, 12, 14, 20, 25, 26}:
            return token == "4"
        elif position in {7}:
            return token == "<pad>"
        elif position in {15, 16, 17, 19, 22, 30}:
            return token == "1"
        elif position in {49, 21, 23, 27, 29}:
            return token == "0"
        elif position in {35, 31}:
            return token == "<s>"
        elif position in {44, 37, 46}:
            return token == "</s>"
        elif position in {40, 41, 42, 43, 45, 47, 48}:
            return token == ""

    attn_0_7_pattern = select_closest(tokens, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0, 40, 44, 45}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2, 3}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {5, 6}:
            return k_position == 10
        elif q_position in {7}:
            return k_position == 12
        elif q_position in {8}:
            return k_position == 14
        elif q_position in {9, 42, 39}:
            return k_position == 16
        elif q_position in {10}:
            return k_position == 18
        elif q_position in {11, 47}:
            return k_position == 19
        elif q_position in {12, 13, 14}:
            return k_position == 20
        elif q_position in {49, 15}:
            return k_position == 24
        elif q_position in {16}:
            return k_position == 25
        elif q_position in {17, 43}:
            return k_position == 26
        elif q_position in {18, 19}:
            return k_position == 27
        elif q_position in {20}:
            return k_position == 28
        elif q_position in {21, 22}:
            return k_position == 30
        elif q_position in {23}:
            return k_position == 36
        elif q_position in {24, 25, 27, 28}:
            return k_position == 37
        elif q_position in {36, 37, 38, 26, 30}:
            return k_position == 41
        elif q_position in {29}:
            return k_position == 46
        elif q_position in {33, 34, 35, 31}:
            return k_position == 42
        elif q_position in {32}:
            return k_position == 49
        elif q_position in {41}:
            return k_position == 17
        elif q_position in {46}:
            return k_position == 8
        elif q_position in {48}:
            return k_position == 21

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0, 12}:
            return k_position == 16
        elif q_position in {1}:
            return k_position == 0
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
        elif q_position in {7}:
            return k_position == 10
        elif q_position in {8}:
            return k_position == 11
        elif q_position in {9}:
            return k_position == 12
        elif q_position in {10}:
            return k_position == 13
        elif q_position in {11}:
            return k_position == 15
        elif q_position in {13}:
            return k_position == 17
        elif q_position in {14}:
            return k_position == 18
        elif q_position in {15}:
            return k_position == 20
        elif q_position in {16}:
            return k_position == 21
        elif q_position in {17}:
            return k_position == 22
        elif q_position in {18}:
            return k_position == 23
        elif q_position in {19}:
            return k_position == 24
        elif q_position in {20}:
            return k_position == 25
        elif q_position in {21}:
            return k_position == 26
        elif q_position in {22, 39}:
            return k_position == 27
        elif q_position in {48, 23}:
            return k_position == 28
        elif q_position in {24, 45}:
            return k_position == 29
        elif q_position in {25, 49}:
            return k_position == 30
        elif q_position in {26}:
            return k_position == 31
        elif q_position in {27}:
            return k_position == 32
        elif q_position in {43, 28}:
            return k_position == 34
        elif q_position in {29, 30}:
            return k_position == 36
        elif q_position in {42, 31}:
            return k_position == 37
        elif q_position in {32, 33, 46}:
            return k_position == 38
        elif q_position in {34}:
            return k_position == 40
        elif q_position in {35, 38}:
            return k_position == 45
        elif q_position in {41, 36}:
            return k_position == 44
        elif q_position in {37}:
            return k_position == 47
        elif q_position in {40}:
            return k_position == 33
        elif q_position in {44}:
            return k_position == 41
        elif q_position in {47}:
            return k_position == 39

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 32, 33, 34, 35, 36, 37, 38, 23, 24, 25, 26, 27}:
            return token == "1"
        elif position in {1, 2, 3, 4, 5, 6, 9, 10, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
            return token == ""
        elif position in {40, 7}:
            return token == "<pad>"
        elif position in {8, 17, 19, 14}:
            return token == "<s>"
        elif position in {11, 12, 13, 15, 16, 18, 20}:
            return token == "</s>"
        elif position in {21, 22}:
            return token == "2"
        elif position in {39, 28, 29, 30, 31}:
            return token == "0"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 2, 3, 4, 5, 6, 7, 8, 9, 39, 40, 44, 47}:
            return token == "1"
        elif position in {1}:
            return token == "</s>"
        elif position in {10, 11, 12}:
            return token == "0"
        elif position in {
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
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            41,
            42,
            43,
            45,
            46,
            48,
            49,
        }:
            return token == ""

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {
            0,
            1,
            2,
            3,
            5,
            6,
            7,
            9,
            10,
            11,
            24,
            25,
            27,
            30,
            36,
            40,
            41,
            42,
            43,
            45,
            46,
            48,
        }:
            return token == ""
        elif position in {49, 4, 44}:
            return token == "<pad>"
        elif position in {8, 12, 37}:
            return token == "<s>"
        elif position in {32, 21, 13}:
            return token == "1"
        elif position in {
            34,
            35,
            38,
            39,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            22,
            23,
            26,
            28,
            31,
        }:
            return token == "0"
        elif position in {33, 29}:
            return token == "</s>"
        elif position in {47}:
            return token == "2"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {0, 33, 32, 35, 31}:
            return token == "2"
        elif position in {
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
            40,
            41,
            42,
            43,
            44,
            46,
            47,
            48,
            49,
        }:
            return token == ""
        elif position in {11, 12, 14, 20, 21, 23, 24, 26}:
            return token == "</s>"
        elif position in {13, 15, 16, 17, 18, 19, 22, 25}:
            return token == "<s>"
        elif position in {27, 28, 29, 30}:
            return token == "4"
        elif position in {34, 38, 39}:
            return token == "0"
        elif position in {45, 36, 37}:
            return token == "1"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {0}:
            return token == "1"
        elif position in {1, 2, 3, 4, 5, 6, 7, 8, 47}:
            return token == "0"
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
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            48,
            49,
        }:
            return token == ""

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {
            0,
            1,
            2,
            3,
            4,
            7,
            9,
            10,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
        }:
            return token == ""
        elif position in {35, 5, 6}:
            return token == "<pad>"
        elif position in {8, 12, 14}:
            return token == "</s>"
        elif position in {11, 13}:
            return token == "<s>"
        elif position in {36, 37, 15, 16, 17, 18, 19, 20, 22, 23, 26, 28, 29, 30, 31}:
            return token == "1"
        elif position in {32, 33, 34, 38, 21, 24, 25, 27}:
            return token == "0"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_4_output, attn_0_6_output):
        key = (attn_0_4_output, attn_0_6_output)
        return 43

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_4_outputs, attn_0_6_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position, attn_0_5_output):
        key = (position, attn_0_5_output)
        if key in {
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "</s>"),
            (1, "<s>"),
            (15, "0"),
            (15, "1"),
            (15, "2"),
            (15, "3"),
            (15, "4"),
            (15, "</s>"),
            (15, "<s>"),
            (16, "0"),
            (16, "1"),
            (16, "2"),
            (16, "3"),
            (16, "4"),
            (16, "</s>"),
            (16, "<s>"),
            (17, "0"),
            (17, "1"),
            (17, "2"),
            (17, "3"),
            (17, "4"),
            (17, "</s>"),
            (17, "<s>"),
            (18, "0"),
            (18, "1"),
            (18, "2"),
            (18, "3"),
            (18, "4"),
            (18, "</s>"),
            (18, "<s>"),
            (40, "0"),
            (40, "1"),
            (40, "3"),
            (40, "<s>"),
            (41, "0"),
            (41, "1"),
            (41, "3"),
            (42, "0"),
            (42, "1"),
            (42, "3"),
            (43, "0"),
            (43, "1"),
            (43, "3"),
            (44, "0"),
            (44, "1"),
            (44, "<s>"),
            (45, "0"),
            (45, "1"),
            (45, "3"),
            (46, "0"),
            (46, "1"),
            (46, "3"),
            (47, "0"),
            (47, "1"),
            (48, "0"),
            (48, "1"),
            (48, "3"),
            (48, "<s>"),
            (49, "0"),
            (49, "1"),
            (49, "3"),
        }:
            return 40
        elif key in {
            (4, "0"),
            (4, "1"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "</s>"),
            (4, "<s>"),
            (5, "0"),
            (5, "1"),
            (5, "2"),
            (5, "3"),
            (5, "4"),
            (5, "</s>"),
            (5, "<s>"),
            (6, "0"),
            (6, "1"),
            (6, "2"),
            (6, "3"),
            (6, "4"),
            (6, "</s>"),
            (6, "<s>"),
            (7, "0"),
            (7, "1"),
            (7, "2"),
            (7, "3"),
            (7, "4"),
            (7, "</s>"),
            (7, "<s>"),
            (8, "0"),
            (8, "1"),
            (8, "2"),
            (8, "3"),
            (8, "4"),
            (8, "</s>"),
            (8, "<s>"),
            (9, "0"),
            (9, "1"),
            (9, "2"),
            (9, "3"),
            (9, "4"),
            (9, "</s>"),
            (9, "<s>"),
            (10, "0"),
            (10, "1"),
            (10, "2"),
            (10, "3"),
            (10, "4"),
            (10, "</s>"),
            (10, "<s>"),
            (11, "0"),
            (11, "1"),
            (11, "2"),
            (11, "3"),
            (11, "4"),
            (11, "</s>"),
            (11, "<s>"),
            (12, "0"),
            (12, "1"),
            (12, "2"),
            (12, "3"),
            (12, "4"),
            (12, "</s>"),
            (12, "<s>"),
            (13, "0"),
            (13, "1"),
            (13, "2"),
            (13, "3"),
            (13, "4"),
            (13, "</s>"),
            (13, "<s>"),
            (14, "0"),
            (14, "1"),
            (14, "2"),
            (14, "3"),
            (14, "4"),
            (14, "</s>"),
            (14, "<s>"),
            (40, "4"),
            (40, "</s>"),
            (41, "2"),
            (41, "4"),
            (41, "</s>"),
            (41, "<s>"),
            (42, "2"),
            (42, "4"),
            (42, "</s>"),
            (42, "<s>"),
            (43, "2"),
            (43, "4"),
            (43, "</s>"),
            (43, "<s>"),
            (44, "4"),
            (44, "</s>"),
            (45, "2"),
            (45, "4"),
            (45, "</s>"),
            (45, "<s>"),
            (46, "2"),
            (46, "4"),
            (46, "</s>"),
            (47, "2"),
            (47, "3"),
            (47, "4"),
            (47, "</s>"),
            (47, "<s>"),
            (48, "2"),
            (48, "4"),
            (48, "</s>"),
            (49, "4"),
            (49, "</s>"),
            (49, "<s>"),
        }:
            return 15
        elif key in {
            (0, "0"),
            (0, "1"),
            (0, "2"),
            (0, "3"),
            (0, "4"),
            (0, "</s>"),
            (0, "<s>"),
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "</s>"),
            (2, "<s>"),
            (3, "0"),
            (3, "1"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (3, "</s>"),
            (3, "<s>"),
            (40, "2"),
            (44, "2"),
            (44, "3"),
            (49, "2"),
        }:
            return 43
        return 24

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(positions, attn_0_5_outputs)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_7_output, attn_0_2_output):
        key = (attn_0_7_output, attn_0_2_output)
        return 41

    mlp_0_2_outputs = [
        mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_7_outputs, attn_0_2_outputs)
    ]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_3_output, position):
        key = (attn_0_3_output, position)
        if key in {
            ("0", 6),
            ("0", 13),
            ("0", 15),
            ("0", 32),
            ("0", 34),
            ("1", 13),
            ("1", 15),
            ("1", 32),
            ("2", 6),
            ("2", 13),
            ("2", 15),
            ("2", 32),
            ("2", 34),
            ("3", 6),
            ("3", 13),
            ("3", 15),
            ("3", 32),
            ("3", 34),
            ("3", 36),
            ("4", 6),
            ("4", 13),
            ("4", 15),
            ("4", 32),
            ("4", 34),
            ("4", 36),
            ("</s>", 6),
            ("</s>", 11),
            ("</s>", 13),
            ("</s>", 15),
            ("</s>", 32),
            ("</s>", 34),
            ("</s>", 36),
            ("<s>", 13),
            ("<s>", 15),
            ("<s>", 32),
        }:
            return 34
        elif key in {
            ("0", 8),
            ("0", 9),
            ("0", 10),
            ("0", 11),
            ("0", 12),
            ("0", 14),
            ("0", 16),
            ("0", 44),
            ("1", 8),
            ("1", 9),
            ("1", 10),
            ("1", 11),
            ("1", 12),
            ("1", 14),
            ("1", 16),
            ("2", 8),
            ("2", 9),
            ("2", 10),
            ("2", 11),
            ("2", 12),
            ("2", 14),
            ("2", 16),
            ("3", 8),
            ("3", 9),
            ("3", 10),
            ("3", 11),
            ("3", 12),
            ("3", 14),
            ("3", 16),
            ("3", 40),
            ("3", 44),
            ("3", 47),
            ("4", 8),
            ("4", 9),
            ("4", 10),
            ("4", 11),
            ("4", 12),
            ("4", 14),
            ("4", 16),
            ("4", 42),
            ("4", 43),
            ("4", 44),
            ("4", 45),
            ("4", 46),
            ("4", 47),
            ("4", 49),
            ("</s>", 8),
            ("</s>", 9),
            ("</s>", 10),
            ("</s>", 12),
            ("</s>", 14),
            ("</s>", 16),
            ("<s>", 8),
            ("<s>", 9),
            ("<s>", 10),
            ("<s>", 11),
            ("<s>", 12),
            ("<s>", 14),
            ("<s>", 16),
        }:
            return 46
        elif key in {
            ("0", 1),
            ("0", 2),
            ("0", 3),
            ("0", 4),
            ("1", 1),
            ("1", 2),
            ("2", 1),
            ("2", 2),
            ("2", 3),
            ("2", 4),
            ("3", 1),
            ("3", 2),
            ("3", 3),
            ("3", 4),
            ("4", 1),
            ("4", 2),
            ("4", 3),
            ("4", 4),
            ("4", 40),
            ("4", 41),
            ("4", 48),
            ("</s>", 1),
            ("</s>", 2),
            ("</s>", 3),
            ("</s>", 4),
            ("<s>", 1),
            ("<s>", 2),
        }:
            return 32
        elif key in {
            ("0", 5),
            ("0", 39),
            ("0", 40),
            ("0", 41),
            ("0", 42),
            ("0", 43),
            ("0", 45),
            ("0", 46),
            ("0", 47),
            ("0", 48),
            ("0", 49),
            ("1", 3),
            ("1", 4),
            ("1", 5),
            ("1", 39),
            ("1", 40),
            ("1", 41),
            ("1", 42),
            ("1", 43),
            ("1", 44),
            ("1", 45),
            ("1", 46),
            ("1", 47),
            ("1", 48),
            ("1", 49),
            ("2", 5),
            ("2", 39),
            ("2", 40),
            ("2", 41),
            ("2", 42),
            ("2", 43),
            ("2", 44),
            ("2", 45),
            ("2", 46),
            ("2", 47),
            ("2", 48),
            ("2", 49),
            ("3", 5),
            ("3", 39),
            ("3", 41),
            ("3", 42),
            ("3", 43),
            ("3", 45),
            ("3", 46),
            ("3", 48),
            ("3", 49),
            ("4", 5),
            ("4", 39),
            ("</s>", 5),
            ("</s>", 39),
            ("</s>", 40),
            ("</s>", 41),
            ("</s>", 42),
            ("</s>", 43),
            ("</s>", 44),
            ("</s>", 45),
            ("</s>", 46),
            ("</s>", 47),
            ("</s>", 48),
            ("</s>", 49),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 5),
            ("<s>", 39),
            ("<s>", 40),
            ("<s>", 41),
            ("<s>", 42),
            ("<s>", 43),
            ("<s>", 44),
            ("<s>", 45),
            ("<s>", 46),
            ("<s>", 47),
            ("<s>", 48),
            ("<s>", 49),
        }:
            return 5
        elif key in {("0", 36), ("1", 6), ("1", 34), ("2", 36)}:
            return 21
        return 10

    mlp_0_3_outputs = [mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_3_outputs, positions)]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output, num_attn_0_5_output):
        key = (num_attn_0_0_output, num_attn_0_5_output)
        return 7

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_5_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_0_output, num_attn_0_6_output):
        key = (num_attn_0_0_output, num_attn_0_6_output)
        return 11

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_2_output, num_attn_0_5_output):
        key = (num_attn_0_2_output, num_attn_0_5_output)
        return 7

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_5_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_3_output, num_attn_0_7_output):
        key = (num_attn_0_3_output, num_attn_0_7_output)
        return 5

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_7_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, token):
        if position in {0, 33, 35, 11, 13, 16}:
            return token == "2"
        elif position in {1, 3}:
            return token == "1"
        elif position in {40, 49, 2}:
            return token == "</s>"
        elif position in {
            4,
            5,
            7,
            8,
            10,
            12,
            14,
            15,
            17,
            18,
            19,
            20,
            21,
            22,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            34,
            36,
            37,
            42,
            44,
            45,
            46,
            47,
        }:
            return token == "4"
        elif position in {6, 41, 43, 48, 23}:
            return token == ""
        elif position in {9}:
            return token == "0"
        elif position in {32, 39, 31}:
            return token == "<s>"
        elif position in {38}:
            return token == "3"

    attn_1_0_pattern = select_closest(tokens, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(position, mlp_0_1_output):
        if position in {0, 24, 11, 41}:
            return mlp_0_1_output == 3
        elif position in {1, 12, 13, 31}:
            return mlp_0_1_output == 43
        elif position in {49, 2, 3, 4}:
            return mlp_0_1_output == 1
        elif position in {34, 5, 44, 18, 26, 28, 29}:
            return mlp_0_1_output == 6
        elif position in {8, 6, 15}:
            return mlp_0_1_output == 0
        elif position in {35, 7}:
            return mlp_0_1_output == 40
        elif position in {9, 46, 47, 19, 21, 23}:
            return mlp_0_1_output == 7
        elif position in {10}:
            return mlp_0_1_output == 29
        elif position in {32, 14, 22}:
            return mlp_0_1_output == 15
        elif position in {38, 16, 20, 25, 27, 30}:
            return mlp_0_1_output == 4
        elif position in {17, 39}:
            return mlp_0_1_output == 2
        elif position in {33}:
            return mlp_0_1_output == 24
        elif position in {36, 37}:
            return mlp_0_1_output == 5
        elif position in {40}:
            return mlp_0_1_output == 13
        elif position in {42}:
            return mlp_0_1_output == 49
        elif position in {43}:
            return mlp_0_1_output == 25
        elif position in {45}:
            return mlp_0_1_output == 42
        elif position in {48}:
            return mlp_0_1_output == 23

    attn_1_1_pattern = select_closest(mlp_0_1_outputs, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_3_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(position, token):
        if position in {
            0,
            5,
            6,
            7,
            8,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            19,
            20,
            21,
            23,
            25,
            33,
            35,
        }:
            return token == "4"
        elif position in {1, 2}:
            return token == "1"
        elif position in {9, 3, 4}:
            return token == "2"
        elif position in {34, 36, 37, 38, 39, 47, 18, 22, 24, 26, 27, 30}:
            return token == "3"
        elif position in {32, 28, 46, 31}:
            return token == "<s>"
        elif position in {41, 42, 43, 44, 29}:
            return token == ""
        elif position in {40, 49, 45}:
            return token == "0"
        elif position in {48}:
            return token == "</s>"

    attn_1_2_pattern = select_closest(tokens, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_3_output, num_mlp_0_3_output):
        if attn_0_3_output in {"0", "1"}:
            return num_mlp_0_3_output == 5
        elif attn_0_3_output in {"3", "2"}:
            return num_mlp_0_3_output == 6
        elif attn_0_3_output in {"4"}:
            return num_mlp_0_3_output == 22
        elif attn_0_3_output in {"</s>"}:
            return num_mlp_0_3_output == 34
        elif attn_0_3_output in {"<s>"}:
            return num_mlp_0_3_output == 45

    attn_1_3_pattern = select_closest(
        num_mlp_0_3_outputs, attn_0_3_outputs, predicate_1_3
    )
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_2_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(attn_0_3_output, attn_0_6_output):
        if attn_0_3_output in {"0", "1", "2"}:
            return attn_0_6_output == ""
        elif attn_0_3_output in {"3", "<s>"}:
            return attn_0_6_output == "2"
        elif attn_0_3_output in {"4"}:
            return attn_0_6_output == "</s>"
        elif attn_0_3_output in {"</s>"}:
            return attn_0_6_output == "<pad>"

    attn_1_4_pattern = select_closest(attn_0_6_outputs, attn_0_3_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_5_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(position, mlp_0_1_output):
        if position in {0, 2, 4, 7, 13}:
            return mlp_0_1_output == 0
        elif position in {1, 43, 6}:
            return mlp_0_1_output == 5
        elif position in {8, 35, 3}:
            return mlp_0_1_output == 7
        elif position in {5}:
            return mlp_0_1_output == 17
        elif position in {32, 9, 10, 17, 26, 28}:
            return mlp_0_1_output == 43
        elif position in {11, 45, 14, 16, 19, 20, 21, 24, 25, 29}:
            return mlp_0_1_output == 15
        elif position in {12, 46}:
            return mlp_0_1_output == 49
        elif position in {15}:
            return mlp_0_1_output == 34
        elif position in {18}:
            return mlp_0_1_output == 22
        elif position in {22}:
            return mlp_0_1_output == 14
        elif position in {27, 23}:
            return mlp_0_1_output == 13
        elif position in {30}:
            return mlp_0_1_output == 21
        elif position in {31}:
            return mlp_0_1_output == 24
        elif position in {33}:
            return mlp_0_1_output == 40
        elif position in {34}:
            return mlp_0_1_output == 6
        elif position in {36, 37, 39}:
            return mlp_0_1_output == 4
        elif position in {38}:
            return mlp_0_1_output == 8
        elif position in {40, 41, 47}:
            return mlp_0_1_output == 2
        elif position in {42}:
            return mlp_0_1_output == 28
        elif position in {44}:
            return mlp_0_1_output == 10
        elif position in {48}:
            return mlp_0_1_output == 3
        elif position in {49}:
            return mlp_0_1_output == 38

    attn_1_5_pattern = select_closest(mlp_0_1_outputs, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_1_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(mlp_0_3_output, num_mlp_0_1_output):
        if mlp_0_3_output in {0, 18, 21, 13}:
            return num_mlp_0_1_output == 2
        elif mlp_0_3_output in {1}:
            return num_mlp_0_1_output == 34
        elif mlp_0_3_output in {2, 4, 6, 11, 27}:
            return num_mlp_0_1_output == 6
        elif mlp_0_3_output in {42, 3}:
            return num_mlp_0_1_output == 40
        elif mlp_0_3_output in {5}:
            return num_mlp_0_1_output == 22
        elif mlp_0_3_output in {8, 26, 7}:
            return num_mlp_0_1_output == 0
        elif mlp_0_3_output in {36, 38, 39, 9, 43, 15, 16, 47, 19, 24, 25, 31}:
            return num_mlp_0_1_output == 5
        elif mlp_0_3_output in {10}:
            return num_mlp_0_1_output == 43
        elif mlp_0_3_output in {40, 41, 12, 48, 28}:
            return num_mlp_0_1_output == 7
        elif mlp_0_3_output in {14}:
            return num_mlp_0_1_output == 25
        elif mlp_0_3_output in {17, 29}:
            return num_mlp_0_1_output == 4
        elif mlp_0_3_output in {49, 20}:
            return num_mlp_0_1_output == 37
        elif mlp_0_3_output in {33, 22}:
            return num_mlp_0_1_output == 28
        elif mlp_0_3_output in {23}:
            return num_mlp_0_1_output == 47
        elif mlp_0_3_output in {30}:
            return num_mlp_0_1_output == 38
        elif mlp_0_3_output in {32}:
            return num_mlp_0_1_output == 11
        elif mlp_0_3_output in {34}:
            return num_mlp_0_1_output == 8
        elif mlp_0_3_output in {35}:
            return num_mlp_0_1_output == 24
        elif mlp_0_3_output in {37}:
            return num_mlp_0_1_output == 21
        elif mlp_0_3_output in {44}:
            return num_mlp_0_1_output == 23
        elif mlp_0_3_output in {45, 46}:
            return num_mlp_0_1_output == 18

    attn_1_6_pattern = select_closest(
        num_mlp_0_1_outputs, mlp_0_3_outputs, predicate_1_6
    )
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_6_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(position, attn_0_3_output):
        if position in {0, 33, 36, 46, 16, 19, 26, 30}:
            return attn_0_3_output == "4"
        elif position in {32, 1, 3, 7, 8}:
            return attn_0_3_output == "<s>"
        elif position in {2, 34, 5, 31}:
            return attn_0_3_output == "</s>"
        elif position in {4, 37, 38, 9, 14, 20, 22}:
            return attn_0_3_output == "2"
        elif position in {35, 6, 39, 40, 41, 15, 18, 21, 23, 24, 25, 27, 28, 29}:
            return attn_0_3_output == "3"
        elif position in {17, 10, 12, 49}:
            return attn_0_3_output == "1"
        elif position in {42, 43, 11, 44, 13, 45, 47, 48}:
            return attn_0_3_output == ""

    attn_1_7_pattern = select_closest(attn_0_3_outputs, positions, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_1_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, attn_0_7_output):
        if position in {
            0,
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
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            48,
        }:
            return attn_0_7_output == ""
        elif position in {1, 3, 4, 17, 18}:
            return attn_0_7_output == "<s>"
        elif position in {2, 6, 7, 8, 10, 11, 12, 46, 49}:
            return attn_0_7_output == "1"
        elif position in {5, 47}:
            return attn_0_7_output == "3"
        elif position in {40, 9, 41, 42, 13, 14, 15, 16}:
            return attn_0_7_output == "0"
        elif position in {19, 20, 21}:
            return attn_0_7_output == "</s>"
        elif position in {43, 44, 45}:
            return attn_0_7_output == "2"

    num_attn_1_0_pattern = select(attn_0_7_outputs, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_7_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, attn_0_5_output):
        if position in {
            0,
            2,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            25,
            26,
            27,
            28,
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
            40,
            41,
            42,
            44,
            45,
            47,
            48,
            49,
        }:
            return attn_0_5_output == ""
        elif position in {1, 4, 6, 7, 8, 9}:
            return attn_0_5_output == "2"
        elif position in {10, 3}:
            return attn_0_5_output == "<s>"
        elif position in {5, 39, 11, 12, 14, 46}:
            return attn_0_5_output == "1"
        elif position in {13}:
            return attn_0_5_output == "0"
        elif position in {15}:
            return attn_0_5_output == "</s>"
        elif position in {24, 43}:
            return attn_0_5_output == "<pad>"

    num_attn_1_1_pattern = select(attn_0_5_outputs, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_2_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(num_mlp_0_0_output, mlp_0_2_output):
        if num_mlp_0_0_output in {0, 6}:
            return mlp_0_2_output == 44
        elif num_mlp_0_0_output in {1, 10}:
            return mlp_0_2_output == 11
        elif num_mlp_0_0_output in {2, 37, 7, 19, 21}:
            return mlp_0_2_output == 33
        elif num_mlp_0_0_output in {34, 3, 30}:
            return mlp_0_2_output == 37
        elif num_mlp_0_0_output in {4}:
            return mlp_0_2_output == 25
        elif num_mlp_0_0_output in {9, 43, 5}:
            return mlp_0_2_output == 38
        elif num_mlp_0_0_output in {8, 41, 48, 14}:
            return mlp_0_2_output == 42
        elif num_mlp_0_0_output in {11, 15}:
            return mlp_0_2_output == 26
        elif num_mlp_0_0_output in {26, 12}:
            return mlp_0_2_output == 27
        elif num_mlp_0_0_output in {13, 39}:
            return mlp_0_2_output == 17
        elif num_mlp_0_0_output in {16}:
            return mlp_0_2_output == 24
        elif num_mlp_0_0_output in {17, 36, 47}:
            return mlp_0_2_output == 16
        elif num_mlp_0_0_output in {32, 49, 18}:
            return mlp_0_2_output == 30
        elif num_mlp_0_0_output in {42, 20}:
            return mlp_0_2_output == 32
        elif num_mlp_0_0_output in {22}:
            return mlp_0_2_output == 47
        elif num_mlp_0_0_output in {23}:
            return mlp_0_2_output == 21
        elif num_mlp_0_0_output in {24, 38}:
            return mlp_0_2_output == 31
        elif num_mlp_0_0_output in {25, 29}:
            return mlp_0_2_output == 14
        elif num_mlp_0_0_output in {27}:
            return mlp_0_2_output == 36
        elif num_mlp_0_0_output in {28}:
            return mlp_0_2_output == 34
        elif num_mlp_0_0_output in {31}:
            return mlp_0_2_output == 12
        elif num_mlp_0_0_output in {33, 46}:
            return mlp_0_2_output == 49
        elif num_mlp_0_0_output in {35}:
            return mlp_0_2_output == 39
        elif num_mlp_0_0_output in {40}:
            return mlp_0_2_output == 8
        elif num_mlp_0_0_output in {44}:
            return mlp_0_2_output == 19
        elif num_mlp_0_0_output in {45}:
            return mlp_0_2_output == 23

    num_attn_1_2_pattern = select(
        mlp_0_2_outputs, num_mlp_0_0_outputs, num_predicate_1_2
    )
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_3_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, token):
        if position in {
            0,
            7,
            8,
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
            44,
            49,
        }:
            return token == ""
        elif position in {1, 10, 11}:
            return token == "<s>"
        elif position in {2, 3, 5}:
            return token == "2"
        elif position in {4}:
            return token == "1"
        elif position in {6, 39, 40, 9, 41, 42, 43, 45, 46, 47, 48}:
            return token == "0"

    num_attn_1_3_pattern = select(tokens, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_3_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(q_position, k_position):
        if q_position in {0}:
            return k_position == 45
        elif q_position in {1}:
            return k_position == 6
        elif q_position in {2, 5, 39, 44, 47, 48, 49, 31}:
            return k_position == 8
        elif q_position in {9, 10, 3, 4}:
            return k_position == 15
        elif q_position in {6, 7}:
            return k_position == 11
        elif q_position in {8}:
            return k_position == 48
        elif q_position in {11, 14}:
            return k_position == 17
        elif q_position in {40, 12, 45, 15}:
            return k_position == 18
        elif q_position in {13}:
            return k_position == 16
        elif q_position in {16}:
            return k_position == 19
        elif q_position in {17}:
            return k_position == 20
        elif q_position in {18}:
            return k_position == 22
        elif q_position in {43, 19, 20}:
            return k_position == 23
        elif q_position in {21}:
            return k_position == 28
        elif q_position in {22}:
            return k_position == 35
        elif q_position in {42, 46, 23}:
            return k_position == 31
        elif q_position in {24, 25}:
            return k_position == 32
        elif q_position in {26, 27, 29}:
            return k_position == 34
        elif q_position in {28}:
            return k_position == 33
        elif q_position in {34, 30}:
            return k_position == 39
        elif q_position in {32}:
            return k_position == 7
        elif q_position in {33}:
            return k_position == 0
        elif q_position in {35, 36, 37}:
            return k_position == 49
        elif q_position in {38}:
            return k_position == 9
        elif q_position in {41}:
            return k_position == 14

    num_attn_1_4_pattern = select(positions, positions, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_4_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(position, token):
        if position in {
            0,
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
            28,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
        }:
            return token == ""
        elif position in {1, 38}:
            return token == "1"
        elif position in {2, 3, 4}:
            return token == "0"
        elif position in {27, 39}:
            return token == "<pad>"
        elif position in {29, 30}:
            return token == "</s>"
        elif position in {32, 33, 34, 35, 31}:
            return token == "4"
        elif position in {36, 37}:
            return token == "2"

    num_attn_1_5_pattern = select(tokens, positions, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_6_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(mlp_0_1_output, attn_0_6_output):
        if mlp_0_1_output in {
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
            16,
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
            39,
            41,
            42,
            44,
            45,
            46,
            47,
            48,
            49,
        }:
            return attn_0_6_output == ""
        elif mlp_0_1_output in {32, 13, 38}:
            return attn_0_6_output == "</s>"
        elif mlp_0_1_output in {40, 17, 43, 14}:
            return attn_0_6_output == "<s>"
        elif mlp_0_1_output in {15}:
            return attn_0_6_output == "1"
        elif mlp_0_1_output in {37}:
            return attn_0_6_output == "<pad>"

    num_attn_1_6_pattern = select(attn_0_6_outputs, mlp_0_1_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_5_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(mlp_0_3_output, token):
        if mlp_0_3_output in {0}:
            return token == "<pad>"
        elif mlp_0_3_output in {1}:
            return token == "</s>"
        elif mlp_0_3_output in {2, 29}:
            return token == "<s>"
        elif mlp_0_3_output in {
            3,
            7,
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
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            45,
            46,
            47,
            48,
            49,
        }:
            return token == ""
        elif mlp_0_3_output in {32, 4, 5, 6, 8, 9, 44}:
            return token == "0"

    num_attn_1_7_pattern = select(tokens, mlp_0_3_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_3_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_4_output, attn_1_2_output):
        key = (attn_0_4_output, attn_1_2_output)
        return 20

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_4_outputs, attn_1_2_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_6_output, attn_1_5_output):
        key = (attn_0_6_output, attn_1_5_output)
        return 42

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_1_5_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(position):
        key = position
        if key in {1, 2}:
            return 35
        elif key in {26, 38}:
            return 36
        return 48

    mlp_1_2_outputs = [mlp_1_2(k0) for k0 in positions]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_1_4_output, position):
        key = (attn_1_4_output, position)
        return 42

    mlp_1_3_outputs = [mlp_1_3(k0, k1) for k0, k1 in zip(attn_1_4_outputs, positions)]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_5_output):
        key = num_attn_1_5_output
        return 1

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_1_5_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_3_output, num_attn_1_3_output):
        key = (num_attn_0_3_output, num_attn_1_3_output)
        return 27

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_0_output, num_attn_1_5_output):
        key = (num_attn_1_0_output, num_attn_1_5_output)
        return 27

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_4_output, num_attn_1_2_output):
        key = (num_attn_1_4_output, num_attn_1_2_output)
        return 16

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_1_1_output, position):
        if attn_1_1_output in {"0"}:
            return position == 35
        elif attn_1_1_output in {"1"}:
            return position == 4
        elif attn_1_1_output in {"2"}:
            return position == 1
        elif attn_1_1_output in {"3"}:
            return position == 39
        elif attn_1_1_output in {"4"}:
            return position == 14
        elif attn_1_1_output in {"<s>", "</s>"}:
            return position == 0

    attn_2_0_pattern = select_closest(positions, attn_1_1_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_7_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "2"
        elif q_token in {"4", "1"}:
            return k_token == ""
        elif q_token in {"2"}:
            return k_token == "4"
        elif q_token in {"3"}:
            return k_token == "0"
        elif q_token in {"<s>", "</s>"}:
            return k_token == "<s>"

    attn_2_1_pattern = select_closest(tokens, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "2"
        elif q_token in {"1"}:
            return k_token == "4"
        elif q_token in {"2"}:
            return k_token == "3"
        elif q_token in {"3"}:
            return k_token == "1"
        elif q_token in {"4"}:
            return k_token == ""
        elif q_token in {"<s>", "</s>"}:
            return k_token == "<s>"

    attn_2_2_pattern = select_closest(tokens, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_5_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 1
        elif q_position in {2, 4}:
            return k_position == 9
        elif q_position in {3}:
            return k_position == 7
        elif q_position in {8, 11, 5}:
            return k_position == 13
        elif q_position in {6}:
            return k_position == 11
        elif q_position in {35, 7}:
            return k_position == 10
        elif q_position in {9, 10}:
            return k_position == 14
        elif q_position in {12, 14, 15, 16, 17}:
            return k_position == 19
        elif q_position in {43, 13}:
            return k_position == 3
        elif q_position in {18, 20}:
            return k_position == 5
        elif q_position in {19, 23}:
            return k_position == 6
        elif q_position in {21}:
            return k_position == 27
        elif q_position in {32, 22}:
            return k_position == 26
        elif q_position in {24}:
            return k_position == 25
        elif q_position in {25}:
            return k_position == 35
        elif q_position in {26, 27, 28, 29, 30}:
            return k_position == 38
        elif q_position in {31}:
            return k_position == 39
        elif q_position in {33}:
            return k_position == 16
        elif q_position in {34}:
            return k_position == 30
        elif q_position in {36, 37, 39, 40, 41, 42, 44, 46, 47, 49}:
            return k_position == 0
        elif q_position in {38}:
            return k_position == 2
        elif q_position in {45}:
            return k_position == 17
        elif q_position in {48}:
            return k_position == 32

    attn_2_3_pattern = select_closest(positions, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_2_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_7_output, mlp_0_1_output):
        if attn_0_7_output in {"0"}:
            return mlp_0_1_output == 7
        elif attn_0_7_output in {"<s>", "2", "3", "1", "4", "</s>"}:
            return mlp_0_1_output == 15

    attn_2_4_pattern = select_closest(mlp_0_1_outputs, attn_0_7_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, tokens)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(q_token, k_token):
        if q_token in {"3", "0"}:
            return k_token == ""
        elif q_token in {"4", "1", "2"}:
            return k_token == "</s>"
        elif q_token in {"<s>", "</s>"}:
            return k_token == "<s>"

    attn_2_5_pattern = select_closest(tokens, tokens, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_5_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(token, mlp_0_3_output):
        if token in {"4", "0"}:
            return mlp_0_3_output == 32
        elif token in {"1"}:
            return mlp_0_3_output == 10
        elif token in {"2"}:
            return mlp_0_3_output == 46
        elif token in {"3"}:
            return mlp_0_3_output == 25
        elif token in {"</s>"}:
            return mlp_0_3_output == 6
        elif token in {"<s>"}:
            return mlp_0_3_output == 7

    attn_2_6_pattern = select_closest(mlp_0_3_outputs, tokens, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_0_2_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_1_1_output, mlp_0_0_output):
        if attn_1_1_output in {"0"}:
            return mlp_0_0_output == 20
        elif attn_1_1_output in {"<s>", "1"}:
            return mlp_0_0_output == 6
        elif attn_1_1_output in {"2"}:
            return mlp_0_0_output == 19
        elif attn_1_1_output in {"3"}:
            return mlp_0_0_output == 2
        elif attn_1_1_output in {"4", "</s>"}:
            return mlp_0_0_output == 7

    attn_2_7_pattern = select_closest(mlp_0_0_outputs, attn_1_1_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_7_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_1_output, attn_0_2_output):
        if attn_1_1_output in {"0", "2", "1", "4", "</s>"}:
            return attn_0_2_output == ""
        elif attn_1_1_output in {"3"}:
            return attn_0_2_output == "1"
        elif attn_1_1_output in {"<s>"}:
            return attn_0_2_output == "<pad>"

    num_attn_2_0_pattern = select(attn_0_2_outputs, attn_1_1_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, token):
        if position in {0, 1, 4, 13}:
            return token == "</s>"
        elif position in {2, 44}:
            return token == "<s>"
        elif position in {3, 22}:
            return token == "<pad>"
        elif position in {40, 9, 48, 5}:
            return token == "0"
        elif position in {10, 43, 6}:
            return token == "1"
        elif position in {
            7,
            8,
            11,
            12,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            23,
            24,
            25,
            26,
            27,
            28,
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
            41,
            42,
            45,
            46,
            47,
            49,
        }:
            return token == ""

    num_attn_2_1_pattern = select(tokens, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_4_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_4_output, num_mlp_1_0_output):
        if attn_1_4_output in {"0"}:
            return num_mlp_1_0_output == 17
        elif attn_1_4_output in {"1"}:
            return num_mlp_1_0_output == 18
        elif attn_1_4_output in {"2"}:
            return num_mlp_1_0_output == 41
        elif attn_1_4_output in {"3"}:
            return num_mlp_1_0_output == 7
        elif attn_1_4_output in {"4"}:
            return num_mlp_1_0_output == 28
        elif attn_1_4_output in {"</s>"}:
            return num_mlp_1_0_output == 43
        elif attn_1_4_output in {"<s>"}:
            return num_mlp_1_0_output == 48

    num_attn_2_2_pattern = select(
        num_mlp_1_0_outputs, attn_1_4_outputs, num_predicate_2_2
    )
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_5_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(mlp_1_3_output, num_mlp_0_3_output):
        if mlp_1_3_output in {0}:
            return num_mlp_0_3_output == 23
        elif mlp_1_3_output in {1}:
            return num_mlp_0_3_output == 37
        elif mlp_1_3_output in {2, 38, 43, 12, 22}:
            return num_mlp_0_3_output == 36
        elif mlp_1_3_output in {34, 3, 4}:
            return num_mlp_0_3_output == 9
        elif mlp_1_3_output in {5, 46}:
            return num_mlp_0_3_output == 3
        elif mlp_1_3_output in {6}:
            return num_mlp_0_3_output == 34
        elif mlp_1_3_output in {44, 7}:
            return num_mlp_0_3_output == 26
        elif mlp_1_3_output in {8}:
            return num_mlp_0_3_output == 25
        elif mlp_1_3_output in {9}:
            return num_mlp_0_3_output == 41
        elif mlp_1_3_output in {33, 10, 35, 45}:
            return num_mlp_0_3_output == 47
        elif mlp_1_3_output in {24, 11}:
            return num_mlp_0_3_output == 11
        elif mlp_1_3_output in {13}:
            return num_mlp_0_3_output == 31
        elif mlp_1_3_output in {14, 39}:
            return num_mlp_0_3_output == 18
        elif mlp_1_3_output in {15}:
            return num_mlp_0_3_output == 20
        elif mlp_1_3_output in {16, 26}:
            return num_mlp_0_3_output == 42
        elif mlp_1_3_output in {17, 23}:
            return num_mlp_0_3_output == 40
        elif mlp_1_3_output in {18}:
            return num_mlp_0_3_output == 48
        elif mlp_1_3_output in {19}:
            return num_mlp_0_3_output == 35
        elif mlp_1_3_output in {20}:
            return num_mlp_0_3_output == 15
        elif mlp_1_3_output in {29, 37, 21}:
            return num_mlp_0_3_output == 24
        elif mlp_1_3_output in {25, 49}:
            return num_mlp_0_3_output == 32
        elif mlp_1_3_output in {27}:
            return num_mlp_0_3_output == 46
        elif mlp_1_3_output in {28}:
            return num_mlp_0_3_output == 17
        elif mlp_1_3_output in {30}:
            return num_mlp_0_3_output == 38
        elif mlp_1_3_output in {36, 47, 31}:
            return num_mlp_0_3_output == 12
        elif mlp_1_3_output in {32, 41}:
            return num_mlp_0_3_output == 13
        elif mlp_1_3_output in {40}:
            return num_mlp_0_3_output == 30
        elif mlp_1_3_output in {42}:
            return num_mlp_0_3_output == 29
        elif mlp_1_3_output in {48}:
            return num_mlp_0_3_output == 27

    num_attn_2_3_pattern = select(
        num_mlp_0_3_outputs, mlp_1_3_outputs, num_predicate_2_3
    )
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_4_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_1_4_output, attn_1_2_output):
        if attn_1_4_output in {"0"}:
            return attn_1_2_output == "0"
        elif attn_1_4_output in {"3", "<s>", "1", "2"}:
            return attn_1_2_output == ""
        elif attn_1_4_output in {"4"}:
            return attn_1_2_output == "<s>"
        elif attn_1_4_output in {"</s>"}:
            return attn_1_2_output == "</s>"

    num_attn_2_4_pattern = select(attn_1_2_outputs, attn_1_4_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_4_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_1_3_output, attn_1_0_output):
        if attn_1_3_output in {"<s>", "0", "</s>"}:
            return attn_1_0_output == "1"
        elif attn_1_3_output in {"1"}:
            return attn_1_0_output == "0"
        elif attn_1_3_output in {"3", "2"}:
            return attn_1_0_output == "</s>"
        elif attn_1_3_output in {"4"}:
            return attn_1_0_output == ""

    num_attn_2_5_pattern = select(attn_1_0_outputs, attn_1_3_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_4_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_1_output, attn_1_2_output):
        if attn_1_1_output in {"<s>", "0", "</s>", "1"}:
            return attn_1_2_output == "2"
        elif attn_1_1_output in {"2"}:
            return attn_1_2_output == "1"
        elif attn_1_1_output in {"3", "4"}:
            return attn_1_2_output == ""

    num_attn_2_6_pattern = select(attn_1_2_outputs, attn_1_1_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_2_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_1_4_output, attn_1_0_output):
        if attn_1_4_output in {"0"}:
            return attn_1_0_output == "<s>"
        elif attn_1_4_output in {"<s>", "2", "3", "1", "</s>"}:
            return attn_1_0_output == "2"
        elif attn_1_4_output in {"4"}:
            return attn_1_0_output == ""

    num_attn_2_7_pattern = select(attn_1_0_outputs, attn_1_4_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_5_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(num_mlp_0_2_output, attn_2_7_output):
        key = (num_mlp_0_2_output, attn_2_7_output)
        if key in {
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (5, "1"),
            (5, "2"),
            (5, "3"),
            (7, "0"),
            (7, "1"),
            (7, "2"),
            (7, "3"),
            (7, "4"),
            (7, "</s>"),
            (7, "<s>"),
            (8, "0"),
            (8, "1"),
            (8, "2"),
            (8, "3"),
            (9, "0"),
            (9, "1"),
            (9, "2"),
            (9, "3"),
            (14, "0"),
            (14, "1"),
            (14, "2"),
            (14, "3"),
            (14, "4"),
            (14, "</s>"),
            (15, "1"),
            (15, "2"),
            (15, "3"),
            (20, "1"),
            (20, "3"),
            (21, "0"),
            (21, "1"),
            (21, "2"),
            (21, "3"),
            (22, "0"),
            (22, "1"),
            (22, "2"),
            (22, "3"),
            (23, "1"),
            (23, "2"),
            (23, "3"),
            (24, "1"),
            (24, "3"),
            (28, "1"),
            (28, "2"),
            (28, "3"),
            (30, "1"),
            (30, "3"),
            (32, "1"),
            (32, "2"),
            (32, "3"),
            (32, "</s>"),
            (34, "0"),
            (34, "1"),
            (34, "2"),
            (34, "3"),
            (36, "1"),
            (36, "3"),
            (37, "1"),
            (37, "3"),
            (40, "1"),
            (40, "3"),
            (42, "1"),
            (42, "3"),
            (43, "1"),
            (43, "3"),
            (47, "1"),
            (47, "2"),
            (47, "3"),
            (48, "0"),
            (48, "1"),
            (48, "2"),
            (48, "3"),
            (48, "<s>"),
            (49, "0"),
            (49, "1"),
            (49, "2"),
            (49, "3"),
        }:
            return 2
        return 9

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(num_mlp_0_2_outputs, attn_2_7_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_1_1_output, attn_2_1_output):
        key = (attn_1_1_output, attn_2_1_output)
        return 38

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_2_1_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(mlp_1_3_output, mlp_0_3_output):
        key = (mlp_1_3_output, mlp_0_3_output)
        return 24

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(mlp_1_3_outputs, mlp_0_3_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(attn_1_6_output, attn_2_4_output):
        key = (attn_1_6_output, attn_2_4_output)
        return 11

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(attn_1_6_outputs, attn_2_4_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_0_output, num_attn_2_3_output):
        key = (num_attn_2_0_output, num_attn_2_3_output)
        return 49

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_0_outputs, num_attn_2_3_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_2_output, num_attn_1_2_output):
        key = (num_attn_2_2_output, num_attn_1_2_output)
        return 25

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_2_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_2_2_output, num_attn_1_5_output):
        key = (num_attn_2_2_output, num_attn_1_5_output)
        return 47

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_2_2_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_1_6_output, num_attn_1_5_output):
        key = (num_attn_1_6_output, num_attn_1_5_output)
        return 16

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_1_5_outputs)
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


print(
    run(
        [
            "<s>",
            "1",
            "3",
            "0",
            "0",
            "0",
            "3",
            "2",
            "3",
            "1",
            "1",
            "2",
            "0",
            "4",
            "4",
            "0",
            "2",
            "1",
            "2",
            "2",
            "2",
            "4",
            "1",
            "3",
            "2",
            "0",
            "</s>",
        ]
    )
)
