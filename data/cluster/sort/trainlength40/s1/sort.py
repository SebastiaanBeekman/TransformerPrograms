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
        "output/length/rasp/sort/trainlength40/s1/sort_weights.csv",
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
    def predicate_0_0(token, position):
        if token in {"0"}:
            return position == 10
        elif token in {"1"}:
            return position == 8
        elif token in {"2"}:
            return position == 11
        elif token in {"3"}:
            return position == 4
        elif token in {"</s>", "4"}:
            return position == 5
        elif token in {"<s>"}:
            return position == 3

    attn_0_0_pattern = select_closest(positions, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {48, 3}:
            return k_position == 4
        elif q_position in {16, 4, 37, 47}:
            return k_position == 5
        elif q_position in {21, 5}:
            return k_position == 10
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8, 42}:
            return k_position == 9
        elif q_position in {40, 9}:
            return k_position == 11
        elif q_position in {10}:
            return k_position == 13
        elif q_position in {32, 11}:
            return k_position == 12
        elif q_position in {19, 12, 45, 38}:
            return k_position == 15
        elif q_position in {35, 13}:
            return k_position == 16
        elif q_position in {14}:
            return k_position == 17
        elif q_position in {22, 15}:
            return k_position == 21
        elif q_position in {17, 36, 31}:
            return k_position == 19
        elif q_position in {18}:
            return k_position == 20
        elif q_position in {20, 23}:
            return k_position == 24
        elif q_position in {24, 34}:
            return k_position == 25
        elif q_position in {25, 41, 39}:
            return k_position == 14
        elif q_position in {26}:
            return k_position == 28
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 36
        elif q_position in {29}:
            return k_position == 22
        elif q_position in {46, 30}:
            return k_position == 6
        elif q_position in {33}:
            return k_position == 35
        elif q_position in {43}:
            return k_position == 44
        elif q_position in {44}:
            return k_position == 18
        elif q_position in {49}:
            return k_position == 0

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 4}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 8
        elif q_position in {41, 3, 44}:
            return k_position == 9
        elif q_position in {42, 12, 5}:
            return k_position == 7
        elif q_position in {40, 46, 18, 6}:
            return k_position == 10
        elif q_position in {7}:
            return k_position == 5
        elif q_position in {8}:
            return k_position == 13
        elif q_position in {9}:
            return k_position == 6
        elif q_position in {48, 10, 19, 36}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 1
        elif q_position in {13}:
            return k_position == 4
        elif q_position in {25, 14}:
            return k_position == 16
        elif q_position in {33, 21, 15}:
            return k_position == 14
        elif q_position in {16, 17}:
            return k_position == 15
        elif q_position in {20, 28}:
            return k_position == 26
        elif q_position in {22}:
            return k_position == 28
        elif q_position in {23}:
            return k_position == 21
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {26}:
            return k_position == 24
        elif q_position in {35, 27}:
            return k_position == 25
        elif q_position in {43, 29}:
            return k_position == 20
        elif q_position in {30}:
            return k_position == 27
        elif q_position in {31}:
            return k_position == 17
        elif q_position in {32, 34}:
            return k_position == 30
        elif q_position in {45, 37}:
            return k_position == 0
        elif q_position in {38}:
            return k_position == 33
        elif q_position in {39}:
            return k_position == 22
        elif q_position in {47}:
            return k_position == 42
        elif q_position in {49}:
            return k_position == 46

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 9}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2, 10, 37}:
            return k_position == 4
        elif q_position in {42, 3}:
            return k_position == 5
        elif q_position in {4, 13, 30}:
            return k_position == 7
        elif q_position in {5, 14}:
            return k_position == 9
        elif q_position in {36, 6}:
            return k_position == 11
        elif q_position in {7}:
            return k_position == 10
        elif q_position in {8, 22}:
            return k_position == 12
        elif q_position in {17, 18, 11, 34}:
            return k_position == 6
        elif q_position in {24, 12}:
            return k_position == 14
        elif q_position in {15}:
            return k_position == 19
        elif q_position in {16}:
            return k_position == 29
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {40, 20}:
            return k_position == 30
        elif q_position in {21}:
            return k_position == 24
        elif q_position in {31, 23}:
            return k_position == 26
        elif q_position in {25}:
            return k_position == 20
        elif q_position in {26}:
            return k_position == 32
        elif q_position in {27, 45, 47}:
            return k_position == 38
        elif q_position in {28}:
            return k_position == 31
        elif q_position in {29}:
            return k_position == 16
        elif q_position in {32}:
            return k_position == 17
        elif q_position in {33}:
            return k_position == 35
        elif q_position in {35}:
            return k_position == 22
        elif q_position in {38}:
            return k_position == 28
        elif q_position in {39}:
            return k_position == 13
        elif q_position in {41}:
            return k_position == 46
        elif q_position in {43}:
            return k_position == 39
        elif q_position in {44}:
            return k_position == 41
        elif q_position in {46}:
            return k_position == 48
        elif q_position in {48}:
            return k_position == 40
        elif q_position in {49}:
            return k_position == 8

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 27}:
            return k_position == 31
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 46}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 8
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {8, 42, 5, 47}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 9
        elif q_position in {9, 7}:
            return k_position == 10
        elif q_position in {40, 10}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 12
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 14
        elif q_position in {14}:
            return k_position == 15
        elif q_position in {48, 49, 44, 15}:
            return k_position == 16
        elif q_position in {16, 39}:
            return k_position == 17
        elif q_position in {17}:
            return k_position == 18
        elif q_position in {18}:
            return k_position == 19
        elif q_position in {19, 21}:
            return k_position == 23
        elif q_position in {20}:
            return k_position == 21
        elif q_position in {22}:
            return k_position == 24
        elif q_position in {26, 23}:
            return k_position == 27
        elif q_position in {24}:
            return k_position == 26
        elif q_position in {25}:
            return k_position == 29
        elif q_position in {28}:
            return k_position == 30
        elif q_position in {36, 29, 31}:
            return k_position == 37
        elif q_position in {30}:
            return k_position == 32
        elif q_position in {32}:
            return k_position == 33
        elif q_position in {33}:
            return k_position == 34
        elif q_position in {34}:
            return k_position == 35
        elif q_position in {41, 35}:
            return k_position == 36
        elif q_position in {37}:
            return k_position == 38
        elif q_position in {38}:
            return k_position == 39
        elif q_position in {43}:
            return k_position == 43
        elif q_position in {45}:
            return k_position == 25

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0, 43, 44}:
            return k_position == 11
        elif q_position in {1, 11}:
            return k_position == 2
        elif q_position in {2, 3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 28
        elif q_position in {13, 5, 6}:
            return k_position == 8
        elif q_position in {24, 7}:
            return k_position == 6
        elif q_position in {8, 10}:
            return k_position == 7
        elif q_position in {9, 39}:
            return k_position == 3
        elif q_position in {12}:
            return k_position == 17
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {47, 15}:
            return k_position == 10
        elif q_position in {16, 49}:
            return k_position == 18
        elif q_position in {17}:
            return k_position == 9
        elif q_position in {18}:
            return k_position == 14
        elif q_position in {19, 21}:
            return k_position == 20
        elif q_position in {20, 46}:
            return k_position == 22
        elif q_position in {40, 22}:
            return k_position == 23
        elif q_position in {23}:
            return k_position == 12
        elif q_position in {25}:
            return k_position == 27
        elif q_position in {48, 41, 26}:
            return k_position == 5
        elif q_position in {27, 28, 45}:
            return k_position == 29
        elif q_position in {35, 29}:
            return k_position == 34
        elif q_position in {30}:
            return k_position == 37
        elif q_position in {31}:
            return k_position == 32
        elif q_position in {32}:
            return k_position == 25
        elif q_position in {33, 42}:
            return k_position == 21
        elif q_position in {34}:
            return k_position == 38
        elif q_position in {36, 38}:
            return k_position == 39
        elif q_position in {37}:
            return k_position == 0

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {16, 4}:
            return k_position == 6
        elif q_position in {33, 5}:
            return k_position == 9
        elif q_position in {6}:
            return k_position == 12
        elif q_position in {46, 7}:
            return k_position == 13
        elif q_position in {8, 31}:
            return k_position == 11
        elif q_position in {40, 9, 12}:
            return k_position == 5
        elif q_position in {10, 42}:
            return k_position == 17
        elif q_position in {35, 11}:
            return k_position == 14
        elif q_position in {13}:
            return k_position == 18
        elif q_position in {14}:
            return k_position == 19
        elif q_position in {15}:
            return k_position == 7
        elif q_position in {17, 19}:
            return k_position == 21
        elif q_position in {18}:
            return k_position == 22
        elif q_position in {20}:
            return k_position == 27
        elif q_position in {21}:
            return k_position == 28
        elif q_position in {24, 22}:
            return k_position == 29
        elif q_position in {23}:
            return k_position == 8
        elif q_position in {25}:
            return k_position == 32
        elif q_position in {26, 29}:
            return k_position == 31
        elif q_position in {27, 30}:
            return k_position == 34
        elif q_position in {34, 28, 45}:
            return k_position == 33
        elif q_position in {32, 36}:
            return k_position == 37
        elif q_position in {37, 38}:
            return k_position == 39
        elif q_position in {39}:
            return k_position == 35
        elif q_position in {41}:
            return k_position == 49
        elif q_position in {43}:
            return k_position == 46
        elif q_position in {44}:
            return k_position == 26
        elif q_position in {47}:
            return k_position == 20
        elif q_position in {48}:
            return k_position == 10
        elif q_position in {49}:
            return k_position == 42

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(position, token):
        if position in {0, 4, 7, 8, 9, 10, 11, 12, 13, 14, 29}:
            return token == "2"
        elif position in {32, 1, 35, 36, 15, 20, 22, 30}:
            return token == "4"
        elif position in {2, 3}:
            return token == "1"
        elif position in {5, 6, 16, 19, 23}:
            return token == "3"
        elif position in {
            17,
            18,
            21,
            25,
            26,
            27,
            28,
            31,
            33,
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
        elif position in {24, 39}:
            return token == "0"
        elif position in {34, 37}:
            return token == "<pad>"
        elif position in {38}:
            return token == "</s>"

    attn_0_7_pattern = select_closest(tokens, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0, 10}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 14
        elif q_position in {2, 3}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 9
        elif q_position in {7}:
            return k_position == 10
        elif q_position in {8}:
            return k_position == 11
        elif q_position in {9}:
            return k_position == 13
        elif q_position in {11}:
            return k_position == 16
        elif q_position in {12}:
            return k_position == 17
        elif q_position in {13}:
            return k_position == 18
        elif q_position in {14}:
            return k_position == 19
        elif q_position in {40, 39, 15}:
            return k_position == 20
        elif q_position in {16, 44}:
            return k_position == 22
        elif q_position in {17, 41}:
            return k_position == 23
        elif q_position in {18, 47}:
            return k_position == 25
        elif q_position in {42, 19}:
            return k_position == 26
        elif q_position in {20}:
            return k_position == 27
        elif q_position in {21}:
            return k_position == 28
        elif q_position in {22}:
            return k_position == 29
        elif q_position in {23}:
            return k_position == 30
        elif q_position in {24, 25}:
            return k_position == 32
        elif q_position in {26, 46}:
            return k_position == 33
        elif q_position in {27}:
            return k_position == 34
        elif q_position in {28}:
            return k_position == 36
        elif q_position in {29, 30}:
            return k_position == 37
        elif q_position in {38, 31}:
            return k_position == 47
        elif q_position in {32}:
            return k_position == 48
        elif q_position in {48, 33}:
            return k_position == 43
        elif q_position in {34}:
            return k_position == 45
        elif q_position in {35}:
            return k_position == 46
        elif q_position in {36}:
            return k_position == 44
        elif q_position in {37}:
            return k_position == 41
        elif q_position in {43}:
            return k_position == 39
        elif q_position in {45}:
            return k_position == 35
        elif q_position in {49}:
            return k_position == 21

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 33, 35, 37, 16, 19, 24, 28, 30, 31}:
            return token == "0"
        elif position in {
            1,
            2,
            3,
            4,
            7,
            8,
            10,
            11,
            27,
            29,
            34,
            38,
            39,
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
        elif position in {9, 45, 5}:
            return token == "<pad>"
        elif position in {6, 12, 13, 18, 20}:
            return token == "</s>"
        elif position in {32, 36, 14, 15, 17, 21, 23, 25, 26}:
            return token == "<s>"
        elif position in {22}:
            return token == "1"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {
            0,
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
        }:
            return token == "0"
        elif position in {1, 2, 3, 4, 7, 8, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49}:
            return token == ""
        elif position in {9, 5}:
            return token == "<pad>"
        elif position in {6, 10, 11, 14, 15, 17, 18, 19}:
            return token == "<s>"
        elif position in {16, 12, 13}:
            return token == "</s>"
        elif position in {20}:
            return token == "2"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0, 12}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {2, 11}:
            return k_position == 14
        elif q_position in {3}:
            return k_position == 7
        elif q_position in {4, 7}:
            return k_position == 9
        elif q_position in {5, 6}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 10
        elif q_position in {9}:
            return k_position == 11
        elif q_position in {10}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 16
        elif q_position in {14}:
            return k_position == 17
        elif q_position in {15}:
            return k_position == 18
        elif q_position in {16}:
            return k_position == 20
        elif q_position in {17}:
            return k_position == 21
        elif q_position in {18}:
            return k_position == 22
        elif q_position in {19}:
            return k_position == 23
        elif q_position in {20}:
            return k_position == 24
        elif q_position in {21}:
            return k_position == 26
        elif q_position in {43, 22, 23}:
            return k_position == 27
        elif q_position in {24, 25}:
            return k_position == 29
        elif q_position in {26, 27}:
            return k_position == 31
        elif q_position in {28}:
            return k_position == 33
        elif q_position in {41, 29, 30}:
            return k_position == 34
        elif q_position in {32, 31}:
            return k_position == 37
        elif q_position in {33, 34}:
            return k_position == 38
        elif q_position in {35}:
            return k_position == 39
        elif q_position in {42, 36}:
            return k_position == 48
        elif q_position in {37}:
            return k_position == 40
        elif q_position in {38}:
            return k_position == 47
        elif q_position in {39}:
            return k_position == 19
        elif q_position in {40, 48, 44, 46}:
            return k_position == 45
        elif q_position in {45}:
            return k_position == 28
        elif q_position in {47}:
            return k_position == 43
        elif q_position in {49}:
            return k_position == 35

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(q_position, k_position):
        if q_position in {0, 8}:
            return k_position == 14
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2, 18}:
            return k_position == 27
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {5, 47}:
            return k_position == 9
        elif q_position in {6}:
            return k_position == 11
        elif q_position in {7}:
            return k_position == 13
        elif q_position in {9}:
            return k_position == 15
        elif q_position in {10, 39}:
            return k_position == 17
        elif q_position in {11}:
            return k_position == 18
        elif q_position in {12}:
            return k_position == 19
        elif q_position in {40, 13}:
            return k_position == 21
        elif q_position in {14}:
            return k_position == 22
        elif q_position in {15}:
            return k_position == 23
        elif q_position in {16}:
            return k_position == 25
        elif q_position in {17}:
            return k_position == 26
        elif q_position in {19}:
            return k_position == 29
        elif q_position in {20}:
            return k_position == 30
        elif q_position in {21}:
            return k_position == 31
        elif q_position in {22}:
            return k_position == 32
        elif q_position in {23}:
            return k_position == 33
        elif q_position in {24, 25}:
            return k_position == 35
        elif q_position in {26}:
            return k_position == 37
        elif q_position in {27}:
            return k_position == 38
        elif q_position in {28}:
            return k_position == 39
        elif q_position in {29, 31}:
            return k_position == 40
        elif q_position in {32, 30}:
            return k_position == 42
        elif q_position in {33}:
            return k_position == 44
        elif q_position in {34, 37}:
            return k_position == 47
        elif q_position in {35}:
            return k_position == 45
        elif q_position in {36}:
            return k_position == 41
        elif q_position in {38}:
            return k_position == 48
        elif q_position in {41, 43, 46, 48, 49}:
            return k_position == 3
        elif q_position in {42}:
            return k_position == 2
        elif q_position in {44, 45}:
            return k_position == 12

    num_attn_0_4_pattern = select(positions, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(q_position, k_position):
        if q_position in {0, 5}:
            return k_position == 13
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 6
        elif q_position in {3}:
            return k_position == 8
        elif q_position in {4}:
            return k_position == 11
        elif q_position in {6}:
            return k_position == 15
        elif q_position in {7}:
            return k_position == 16
        elif q_position in {8, 42}:
            return k_position == 17
        elif q_position in {9, 26}:
            return k_position == 18
        elif q_position in {10, 46}:
            return k_position == 21
        elif q_position in {11}:
            return k_position == 22
        elif q_position in {12}:
            return k_position == 23
        elif q_position in {40, 13}:
            return k_position == 25
        elif q_position in {14, 15}:
            return k_position == 26
        elif q_position in {16}:
            return k_position == 29
        elif q_position in {17, 18}:
            return k_position == 32
        elif q_position in {19, 20}:
            return k_position == 33
        elif q_position in {21}:
            return k_position == 34
        elif q_position in {22}:
            return k_position == 38
        elif q_position in {23}:
            return k_position == 37
        elif q_position in {24, 38, 44, 30}:
            return k_position == 45
        elif q_position in {48, 25, 29}:
            return k_position == 40
        elif q_position in {35, 27}:
            return k_position == 43
        elif q_position in {32, 28, 37}:
            return k_position == 47
        elif q_position in {49, 34, 31}:
            return k_position == 44
        elif q_position in {33}:
            return k_position == 41
        elif q_position in {36}:
            return k_position == 42
        elif q_position in {43, 39}:
            return k_position == 2
        elif q_position in {41}:
            return k_position == 20
        elif q_position in {45}:
            return k_position == 49
        elif q_position in {47}:
            return k_position == 36

    num_attn_0_5_pattern = select(positions, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {0}:
            return token == "2"
        elif position in {1, 2, 3, 4, 40, 41, 42, 44, 45, 46, 47, 48, 49}:
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
            25,
            27,
            28,
            29,
            30,
            31,
            32,
            34,
            35,
            36,
            37,
            38,
            43,
        }:
            return token == ""
        elif position in {24, 33, 26}:
            return token == "<pad>"
        elif position in {39}:
            return token == "1"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {0, 1}:
            return token == "1"
        elif position in {2, 3, 4, 5, 39, 47}:
            return token == "0"
        elif position in {
            6,
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
            20,
            21,
            22,
            23,
            24,
            26,
            27,
            31,
            33,
            34,
            37,
            38,
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
        elif position in {32, 35, 7, 29, 30}:
            return token == "<s>"
        elif position in {14}:
            return token == "<pad>"
        elif position in {25, 28, 36}:
            return token == "</s>"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_1_output, attn_0_6_output):
        key = (attn_0_1_output, attn_0_6_output)
        if key in {
            ("0", "</s>"),
            ("1", "</s>"),
            ("2", "</s>"),
            ("3", "</s>"),
            ("4", "</s>"),
            ("</s>", "</s>"),
            ("<s>", "</s>"),
        }:
            return 6
        return 28

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_6_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(token, attn_0_1_output):
        key = (token, attn_0_1_output)
        return 30

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(tokens, attn_0_1_outputs)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_3_output, position):
        key = (attn_0_3_output, position)
        if key in {
            ("0", 6),
            ("0", 7),
            ("0", 8),
            ("0", 38),
            ("1", 6),
            ("1", 7),
            ("1", 8),
            ("1", 38),
            ("2", 6),
            ("2", 7),
            ("2", 8),
            ("2", 38),
            ("3", 6),
            ("3", 7),
            ("3", 8),
            ("3", 38),
            ("4", 6),
            ("4", 7),
            ("4", 8),
            ("4", 38),
        }:
            return 44
        elif key in {
            ("0", 4),
            ("0", 5),
            ("0", 40),
            ("0", 41),
            ("0", 42),
            ("0", 43),
            ("0", 44),
            ("0", 45),
            ("0", 46),
            ("0", 47),
            ("0", 48),
            ("0", 49),
            ("1", 4),
            ("1", 5),
            ("1", 40),
            ("1", 43),
            ("1", 44),
            ("1", 45),
            ("1", 46),
            ("1", 47),
            ("2", 4),
            ("2", 5),
            ("3", 4),
            ("3", 5),
            ("3", 40),
            ("3", 41),
            ("3", 42),
            ("3", 43),
            ("3", 44),
            ("3", 45),
            ("3", 46),
            ("3", 47),
            ("3", 48),
            ("3", 49),
            ("4", 4),
            ("4", 5),
            ("</s>", 4),
            ("</s>", 5),
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
            ("<s>", 4),
            ("<s>", 5),
            ("<s>", 6),
            ("<s>", 7),
            ("<s>", 8),
            ("<s>", 38),
        }:
            return 24
        elif key in {
            ("0", 1),
            ("0", 2),
            ("0", 39),
            ("1", 1),
            ("1", 2),
            ("1", 39),
            ("2", 1),
            ("2", 2),
            ("2", 39),
            ("3", 1),
            ("3", 2),
            ("3", 39),
            ("4", 1),
            ("4", 2),
            ("4", 39),
            ("</s>", 1),
            ("</s>", 2),
            ("<s>", 1),
            ("<s>", 2),
        }:
            return 10
        elif key in {
            ("0", 3),
            ("1", 3),
            ("2", 3),
            ("3", 3),
            ("4", 3),
            ("</s>", 3),
            ("</s>", 39),
            ("<s>", 39),
        }:
            return 14
        elif key in {
            ("0", 0),
            ("1", 0),
            ("2", 0),
            ("3", 0),
            ("4", 0),
            ("</s>", 0),
            ("<s>", 0),
            ("<s>", 3),
        }:
            return 6
        elif key in {("</s>", 6), ("</s>", 7), ("</s>", 8), ("</s>", 38)}:
            return 35
        return 17

    mlp_0_2_outputs = [mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_3_outputs, positions)]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(position):
        key = position
        if key in {1, 2, 3, 4, 5, 6}:
            return 37
        elif key in {0, 8, 9, 10, 11, 13}:
            return 47
        elif key in {12, 14, 15, 17}:
            return 14
        elif key in {7}:
            return 9
        return 17

    mlp_0_3_outputs = [mlp_0_3(k0) for k0 in positions]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output):
        key = num_attn_0_2_output
        return 4

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_7_output, num_attn_0_5_output):
        key = (num_attn_0_7_output, num_attn_0_5_output)
        return 0

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_7_outputs, num_attn_0_5_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_4_output, num_attn_0_3_output):
        key = (num_attn_0_4_output, num_attn_0_3_output)
        return 17

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_4_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_6_output, num_attn_0_4_output):
        key = (num_attn_0_6_output, num_attn_0_4_output)
        return 21

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_6_outputs, num_attn_0_4_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, mlp_0_3_output):
        if position in {0, 35}:
            return mlp_0_3_output == 4
        elif position in {1, 2, 3, 40, 47, 48, 49, 20}:
            return mlp_0_3_output == 2
        elif position in {10, 4, 5, 7}:
            return mlp_0_3_output == 5
        elif position in {6, 39}:
            return mlp_0_3_output == 3
        elif position in {8, 38, 30, 31}:
            return mlp_0_3_output == 17
        elif position in {37, 9, 43, 16, 21, 23}:
            return mlp_0_3_output == 14
        elif position in {24, 11, 12, 13}:
            return mlp_0_3_output == 37
        elif position in {33, 34, 14, 19, 26, 27}:
            return mlp_0_3_output == 47
        elif position in {32, 15}:
            return mlp_0_3_output == 19
        elif position in {17, 18}:
            return mlp_0_3_output == 9
        elif position in {22}:
            return mlp_0_3_output == 43
        elif position in {25, 42, 28, 36}:
            return mlp_0_3_output == 7
        elif position in {29}:
            return mlp_0_3_output == 48
        elif position in {41, 44, 45}:
            return mlp_0_3_output == 1
        elif position in {46}:
            return mlp_0_3_output == 13

    attn_1_0_pattern = select_closest(mlp_0_3_outputs, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_2_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(mlp_0_2_output, mlp_0_1_output):
        if mlp_0_2_output in {0}:
            return mlp_0_1_output == 42
        elif mlp_0_2_output in {1, 26}:
            return mlp_0_1_output == 31
        elif mlp_0_2_output in {17, 2, 3, 42}:
            return mlp_0_1_output == 30
        elif mlp_0_2_output in {27, 4, 31}:
            return mlp_0_1_output == 9
        elif mlp_0_2_output in {20, 5, 36}:
            return mlp_0_1_output == 27
        elif mlp_0_2_output in {34, 35, 37, 6, 11, 13, 21, 22, 29}:
            return mlp_0_1_output == 6
        elif mlp_0_2_output in {10, 14, 7}:
            return mlp_0_1_output == 39
        elif mlp_0_2_output in {8, 38}:
            return mlp_0_1_output == 0
        elif mlp_0_2_output in {24, 9, 32}:
            return mlp_0_1_output == 5
        elif mlp_0_2_output in {48, 41, 12}:
            return mlp_0_1_output == 40
        elif mlp_0_2_output in {15}:
            return mlp_0_1_output == 21
        elif mlp_0_2_output in {16, 49, 47}:
            return mlp_0_1_output == 37
        elif mlp_0_2_output in {18}:
            return mlp_0_1_output == 11
        elif mlp_0_2_output in {19}:
            return mlp_0_1_output == 10
        elif mlp_0_2_output in {23}:
            return mlp_0_1_output == 4
        elif mlp_0_2_output in {25}:
            return mlp_0_1_output == 20
        elif mlp_0_2_output in {28, 30}:
            return mlp_0_1_output == 25
        elif mlp_0_2_output in {33}:
            return mlp_0_1_output == 32
        elif mlp_0_2_output in {39}:
            return mlp_0_1_output == 44
        elif mlp_0_2_output in {40}:
            return mlp_0_1_output == 3
        elif mlp_0_2_output in {43}:
            return mlp_0_1_output == 14
        elif mlp_0_2_output in {44}:
            return mlp_0_1_output == 46
        elif mlp_0_2_output in {45}:
            return mlp_0_1_output == 33
        elif mlp_0_2_output in {46}:
            return mlp_0_1_output == 7

    attn_1_1_pattern = select_closest(mlp_0_1_outputs, mlp_0_2_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_mlp_0_2_output, k_mlp_0_2_output):
        if q_mlp_0_2_output in {0, 5, 7, 43, 29}:
            return k_mlp_0_2_output == 7
        elif q_mlp_0_2_output in {16, 1, 49}:
            return k_mlp_0_2_output == 44
        elif q_mlp_0_2_output in {33, 2, 3}:
            return k_mlp_0_2_output == 15
        elif q_mlp_0_2_output in {38, 4, 36}:
            return k_mlp_0_2_output == 40
        elif q_mlp_0_2_output in {34, 6, 41, 44, 13, 46, 15, 47}:
            return k_mlp_0_2_output == 6
        elif q_mlp_0_2_output in {8, 10}:
            return k_mlp_0_2_output == 35
        elif q_mlp_0_2_output in {9, 23}:
            return k_mlp_0_2_output == 45
        elif q_mlp_0_2_output in {11}:
            return k_mlp_0_2_output == 42
        elif q_mlp_0_2_output in {12}:
            return k_mlp_0_2_output == 38
        elif q_mlp_0_2_output in {42, 14, 48, 21, 25}:
            return k_mlp_0_2_output == 4
        elif q_mlp_0_2_output in {40, 17}:
            return k_mlp_0_2_output == 17
        elif q_mlp_0_2_output in {18}:
            return k_mlp_0_2_output == 49
        elif q_mlp_0_2_output in {19}:
            return k_mlp_0_2_output == 47
        elif q_mlp_0_2_output in {20}:
            return k_mlp_0_2_output == 36
        elif q_mlp_0_2_output in {22}:
            return k_mlp_0_2_output == 30
        elif q_mlp_0_2_output in {24}:
            return k_mlp_0_2_output == 5
        elif q_mlp_0_2_output in {26, 39}:
            return k_mlp_0_2_output == 3
        elif q_mlp_0_2_output in {27}:
            return k_mlp_0_2_output == 27
        elif q_mlp_0_2_output in {28}:
            return k_mlp_0_2_output == 31
        elif q_mlp_0_2_output in {30}:
            return k_mlp_0_2_output == 33
        elif q_mlp_0_2_output in {35, 31}:
            return k_mlp_0_2_output == 28
        elif q_mlp_0_2_output in {32}:
            return k_mlp_0_2_output == 10
        elif q_mlp_0_2_output in {45, 37}:
            return k_mlp_0_2_output == 2

    attn_1_2_pattern = select_closest(mlp_0_2_outputs, mlp_0_2_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_0_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(mlp_0_2_output, num_mlp_0_0_output):
        if mlp_0_2_output in {0}:
            return num_mlp_0_0_output == 31
        elif mlp_0_2_output in {1}:
            return num_mlp_0_0_output == 32
        elif mlp_0_2_output in {24, 2}:
            return num_mlp_0_0_output == 4
        elif mlp_0_2_output in {3}:
            return num_mlp_0_0_output == 38
        elif mlp_0_2_output in {
            32,
            33,
            35,
            4,
            6,
            38,
            13,
            14,
            15,
            45,
            48,
            20,
            26,
            27,
            29,
        }:
            return num_mlp_0_0_output == 6
        elif mlp_0_2_output in {5}:
            return num_mlp_0_0_output == 44
        elif mlp_0_2_output in {7}:
            return num_mlp_0_0_output == 21
        elif mlp_0_2_output in {8}:
            return num_mlp_0_0_output == 30
        elif mlp_0_2_output in {9}:
            return num_mlp_0_0_output == 39
        elif mlp_0_2_output in {10, 18, 39}:
            return num_mlp_0_0_output == 1
        elif mlp_0_2_output in {34, 11, 46, 25, 30}:
            return num_mlp_0_0_output == 2
        elif mlp_0_2_output in {12}:
            return num_mlp_0_0_output == 40
        elif mlp_0_2_output in {16, 31}:
            return num_mlp_0_0_output == 42
        elif mlp_0_2_output in {17, 43}:
            return num_mlp_0_0_output == 7
        elif mlp_0_2_output in {19}:
            return num_mlp_0_0_output == 26
        elif mlp_0_2_output in {21}:
            return num_mlp_0_0_output == 22
        elif mlp_0_2_output in {22}:
            return num_mlp_0_0_output == 25
        elif mlp_0_2_output in {23}:
            return num_mlp_0_0_output == 19
        elif mlp_0_2_output in {28}:
            return num_mlp_0_0_output == 20
        elif mlp_0_2_output in {36}:
            return num_mlp_0_0_output == 29
        elif mlp_0_2_output in {37}:
            return num_mlp_0_0_output == 8
        elif mlp_0_2_output in {40, 41, 42}:
            return num_mlp_0_0_output == 14
        elif mlp_0_2_output in {44}:
            return num_mlp_0_0_output == 45
        elif mlp_0_2_output in {47}:
            return num_mlp_0_0_output == 23
        elif mlp_0_2_output in {49}:
            return num_mlp_0_0_output == 34

    attn_1_3_pattern = select_closest(
        num_mlp_0_0_outputs, mlp_0_2_outputs, predicate_1_3
    )
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(position, token):
        if position in {0, 1, 2, 40, 26}:
            return token == "1"
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
            19,
            20,
            23,
            24,
            28,
            29,
            31,
            32,
            34,
            35,
            37,
        }:
            return token == "4"
        elif position in {33, 36, 44, 18, 25, 27}:
            return token == "3"
        elif position in {21, 38}:
            return token == "0"
        elif position in {41, 42, 43, 45, 46, 47, 48, 22}:
            return token == ""
        elif position in {30}:
            return token == "</s>"
        elif position in {39}:
            return token == "2"
        elif position in {49}:
            return token == "<s>"

    attn_1_4_pattern = select_closest(tokens, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, tokens)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(q_position, k_position):
        if q_position in {0}:
            return k_position == 20
        elif q_position in {1, 3, 4}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {44, 5}:
            return k_position == 3
        elif q_position in {46, 6}:
            return k_position == 12
        elif q_position in {15, 7}:
            return k_position == 11
        elif q_position in {8}:
            return k_position == 15
        elif q_position in {48, 9, 11, 35}:
            return k_position == 7
        elif q_position in {10, 45}:
            return k_position == 8
        elif q_position in {12}:
            return k_position == 9
        elif q_position in {13}:
            return k_position == 23
        elif q_position in {14, 47}:
            return k_position == 4
        elif q_position in {16, 30}:
            return k_position == 22
        elif q_position in {17}:
            return k_position == 10
        elif q_position in {18, 42}:
            return k_position == 27
        elif q_position in {19}:
            return k_position == 21
        elif q_position in {40, 20}:
            return k_position == 16
        elif q_position in {32, 26, 21, 22}:
            return k_position == 17
        elif q_position in {49, 39, 23}:
            return k_position == 18
        elif q_position in {24}:
            return k_position == 32
        elif q_position in {25}:
            return k_position == 29
        elif q_position in {27}:
            return k_position == 35
        elif q_position in {28}:
            return k_position == 14
        elif q_position in {29, 38}:
            return k_position == 36
        elif q_position in {31}:
            return k_position == 25
        elif q_position in {33}:
            return k_position == 30
        elif q_position in {34}:
            return k_position == 19
        elif q_position in {36}:
            return k_position == 34
        elif q_position in {37}:
            return k_position == 45
        elif q_position in {41, 43}:
            return k_position == 31

    attn_1_5_pattern = select_closest(positions, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_1_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(q_position, k_position):
        if q_position in {0}:
            return k_position == 27
        elif q_position in {1, 4, 37}:
            return k_position == 2
        elif q_position in {2, 26}:
            return k_position == 44
        elif q_position in {3, 5}:
            return k_position == 1
        elif q_position in {6, 43, 48, 23, 30}:
            return k_position == 14
        elif q_position in {7}:
            return k_position == 38
        elif q_position in {8}:
            return k_position == 34
        elif q_position in {9}:
            return k_position == 4
        elif q_position in {35, 40, 10, 44, 14}:
            return k_position == 5
        elif q_position in {11, 12}:
            return k_position == 3
        elif q_position in {33, 36, 13, 39}:
            return k_position == 9
        elif q_position in {15}:
            return k_position == 25
        elif q_position in {16}:
            return k_position == 19
        elif q_position in {17}:
            return k_position == 28
        elif q_position in {18}:
            return k_position == 31
        elif q_position in {19}:
            return k_position == 47
        elif q_position in {20, 29}:
            return k_position == 10
        elif q_position in {42, 45, 46, 47, 21, 28}:
            return k_position == 17
        elif q_position in {24, 22}:
            return k_position == 16
        elif q_position in {25, 34}:
            return k_position == 26
        elif q_position in {27}:
            return k_position == 32
        elif q_position in {31}:
            return k_position == 8
        elif q_position in {32}:
            return k_position == 33
        elif q_position in {38}:
            return k_position == 21
        elif q_position in {41}:
            return k_position == 43
        elif q_position in {49}:
            return k_position == 13

    attn_1_6_pattern = select_closest(positions, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_2_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(mlp_0_3_output, mlp_0_2_output):
        if mlp_0_3_output in {0, 33, 8, 42, 45, 46, 16, 21, 27, 28, 29, 30, 31}:
            return mlp_0_2_output == 5
        elif mlp_0_3_output in {1}:
            return mlp_0_2_output == 32
        elif mlp_0_3_output in {2, 35, 38, 39, 43, 24}:
            return mlp_0_2_output == 17
        elif mlp_0_3_output in {10, 3}:
            return mlp_0_2_output == 24
        elif mlp_0_3_output in {4}:
            return mlp_0_2_output == 45
        elif mlp_0_3_output in {5}:
            return mlp_0_2_output == 6
        elif mlp_0_3_output in {11, 6, 14}:
            return mlp_0_2_output == 0
        elif mlp_0_3_output in {7, 40, 47, 18, 20}:
            return mlp_0_2_output == 14
        elif mlp_0_3_output in {9, 49}:
            return mlp_0_2_output == 29
        elif mlp_0_3_output in {12}:
            return mlp_0_2_output == 47
        elif mlp_0_3_output in {13}:
            return mlp_0_2_output == 2
        elif mlp_0_3_output in {15}:
            return mlp_0_2_output == 27
        elif mlp_0_3_output in {17, 37, 22}:
            return mlp_0_2_output == 44
        elif mlp_0_3_output in {19}:
            return mlp_0_2_output == 43
        elif mlp_0_3_output in {23}:
            return mlp_0_2_output == 23
        elif mlp_0_3_output in {25, 44, 41}:
            return mlp_0_2_output == 48
        elif mlp_0_3_output in {26}:
            return mlp_0_2_output == 4
        elif mlp_0_3_output in {32}:
            return mlp_0_2_output == 3
        elif mlp_0_3_output in {34}:
            return mlp_0_2_output == 34
        elif mlp_0_3_output in {36}:
            return mlp_0_2_output == 49
        elif mlp_0_3_output in {48}:
            return mlp_0_2_output == 21

    attn_1_7_pattern = select_closest(mlp_0_2_outputs, mlp_0_3_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_1_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, token):
        if position in {
            0,
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
            44,
        }:
            return token == ""
        elif position in {16, 1}:
            return token == "<s>"
        elif position in {2, 3, 7, 8, 9, 10, 11, 40, 13, 42, 43, 45, 46, 48, 49}:
            return token == "1"
        elif position in {41, 4, 5, 47}:
            return token == "2"
        elif position in {12, 6}:
            return token == "0"

    num_attn_1_0_pattern = select(tokens, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_1_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, attn_0_5_output):
        if position in {
            0,
            4,
            5,
            13,
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
        }:
            return attn_0_5_output == ""
        elif position in {1}:
            return attn_0_5_output == "2"
        elif position in {2}:
            return attn_0_5_output == "1"
        elif position in {3}:
            return attn_0_5_output == "<s>"
        elif position in {6, 7, 8, 9, 10, 41, 12, 42, 14, 43, 44, 45, 46, 47, 48, 49}:
            return attn_0_5_output == "0"
        elif position in {11}:
            return attn_0_5_output == "<pad>"

    num_attn_1_1_pattern = select(attn_0_5_outputs, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_1_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, token):
        if position in {
            0,
            13,
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
        }:
            return token == ""
        elif position in {1, 14}:
            return token == "<s>"
        elif position in {2, 3, 4, 5, 40, 12, 44, 46, 48, 49}:
            return token == "1"
        elif position in {6, 7, 8, 9, 10, 11, 41, 42, 43, 45, 47}:
            return token == "0"

    num_attn_1_2_pattern = select(tokens, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_1_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, token):
        if position in {0, 6, 7, 8, 39, 41, 42, 43, 44, 45, 15, 46, 47, 48, 49, 22}:
            return token == "0"
        elif position in {40, 1, 21}:
            return token == "2"
        elif position in {2, 4, 37, 11, 19, 24}:
            return token == "<s>"
        elif position in {18, 3}:
            return token == "1"
        elif position in {
            5,
            9,
            10,
            13,
            16,
            20,
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
            38,
        }:
            return token == ""
        elif position in {17, 12, 14}:
            return token == "</s>"

    num_attn_1_3_pattern = select(tokens, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_7_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(position, attn_0_5_output):
        if position in {
            0,
            22,
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
            44,
            46,
            47,
        }:
            return attn_0_5_output == ""
        elif position in {1, 3, 20, 21, 23}:
            return attn_0_5_output == "</s>"
        elif position in {2, 4}:
            return attn_0_5_output == "<s>"
        elif position in {
            5,
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
            40,
            42,
            19,
            43,
            45,
            49,
        }:
            return attn_0_5_output == "2"
        elif position in {48, 41, 6}:
            return attn_0_5_output == "1"
        elif position in {17, 18}:
            return attn_0_5_output == "0"

    num_attn_1_4_pattern = select(attn_0_5_outputs, positions, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_1_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(num_mlp_0_2_output, num_mlp_0_1_output):
        if num_mlp_0_2_output in {0}:
            return num_mlp_0_1_output == 16
        elif num_mlp_0_2_output in {40, 1}:
            return num_mlp_0_1_output == 41
        elif num_mlp_0_2_output in {2}:
            return num_mlp_0_1_output == 38
        elif num_mlp_0_2_output in {3}:
            return num_mlp_0_1_output == 8
        elif num_mlp_0_2_output in {4}:
            return num_mlp_0_1_output == 2
        elif num_mlp_0_2_output in {5}:
            return num_mlp_0_1_output == 40
        elif num_mlp_0_2_output in {34, 6, 17, 20, 29}:
            return num_mlp_0_1_output == 18
        elif num_mlp_0_2_output in {39, 7}:
            return num_mlp_0_1_output == 30
        elif num_mlp_0_2_output in {8}:
            return num_mlp_0_1_output == 13
        elif num_mlp_0_2_output in {9, 12}:
            return num_mlp_0_1_output == 3
        elif num_mlp_0_2_output in {10}:
            return num_mlp_0_1_output == 43
        elif num_mlp_0_2_output in {18, 11, 13}:
            return num_mlp_0_1_output == 4
        elif num_mlp_0_2_output in {14}:
            return num_mlp_0_1_output == 45
        elif num_mlp_0_2_output in {15}:
            return num_mlp_0_1_output == 35
        elif num_mlp_0_2_output in {16, 48}:
            return num_mlp_0_1_output == 28
        elif num_mlp_0_2_output in {27, 35, 19}:
            return num_mlp_0_1_output == 25
        elif num_mlp_0_2_output in {21, 30}:
            return num_mlp_0_1_output == 32
        elif num_mlp_0_2_output in {49, 22}:
            return num_mlp_0_1_output == 23
        elif num_mlp_0_2_output in {23}:
            return num_mlp_0_1_output == 29
        elif num_mlp_0_2_output in {24, 41}:
            return num_mlp_0_1_output == 42
        elif num_mlp_0_2_output in {25}:
            return num_mlp_0_1_output == 14
        elif num_mlp_0_2_output in {26}:
            return num_mlp_0_1_output == 47
        elif num_mlp_0_2_output in {28, 38}:
            return num_mlp_0_1_output == 21
        elif num_mlp_0_2_output in {31}:
            return num_mlp_0_1_output == 44
        elif num_mlp_0_2_output in {32, 43, 36}:
            return num_mlp_0_1_output == 19
        elif num_mlp_0_2_output in {33}:
            return num_mlp_0_1_output == 11
        elif num_mlp_0_2_output in {45, 37}:
            return num_mlp_0_1_output == 7
        elif num_mlp_0_2_output in {42, 46}:
            return num_mlp_0_1_output == 22
        elif num_mlp_0_2_output in {44}:
            return num_mlp_0_1_output == 27
        elif num_mlp_0_2_output in {47}:
            return num_mlp_0_1_output == 49

    num_attn_1_5_pattern = select(
        num_mlp_0_1_outputs, num_mlp_0_2_outputs, num_predicate_1_5
    )
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_7_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(num_mlp_0_3_output, attn_0_7_output):
        if num_mlp_0_3_output in {0, 9, 3}:
            return attn_0_7_output == "</s>"
        elif num_mlp_0_3_output in {1, 33, 8, 12, 44, 16, 48, 19}:
            return attn_0_7_output == "3"
        elif num_mlp_0_3_output in {26, 2, 11, 7}:
            return attn_0_7_output == "<s>"
        elif num_mlp_0_3_output in {
            4,
            37,
            39,
            42,
            14,
            47,
            49,
            18,
            22,
            23,
            25,
            29,
            30,
            31,
        }:
            return attn_0_7_output == "1"
        elif num_mlp_0_3_output in {35, 36, 5, 45, 13}:
            return attn_0_7_output == ""
        elif num_mlp_0_3_output in {34, 6, 38, 10, 46, 17, 21, 24, 27, 28}:
            return attn_0_7_output == "2"
        elif num_mlp_0_3_output in {32, 40, 41, 43, 15, 20}:
            return attn_0_7_output == "0"

    num_attn_1_6_pattern = select(
        attn_0_7_outputs, num_mlp_0_3_outputs, num_predicate_1_6
    )
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_6_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(position, token):
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
            49,
        }:
            return token == ""
        elif position in {1}:
            return token == "<s>"
        elif position in {2, 3, 4, 5, 6, 7, 8, 9, 10, 39, 40, 42, 46, 47}:
            return token == "1"
        elif position in {48, 43, 44}:
            return token == "2"
        elif position in {45}:
            return token == "0"

    num_attn_1_7_pattern = select(tokens, positions, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_3_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_5_output):
        key = attn_1_5_output
        if key in {"", "0"}:
            return 8
        return 23

    mlp_1_0_outputs = [mlp_1_0(k0) for k0 in attn_1_5_outputs]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_5_output, attn_1_1_output):
        key = (attn_1_5_output, attn_1_1_output)
        if key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "2"),
            ("0", "3"),
            ("0", "</s>"),
            ("0", "<s>"),
            ("1", "0"),
            ("1", "2"),
            ("1", "3"),
            ("1", "</s>"),
            ("1", "<s>"),
            ("2", "0"),
            ("2", "2"),
            ("2", "3"),
            ("2", "</s>"),
            ("2", "<s>"),
            ("3", "<s>"),
            ("</s>", "0"),
            ("</s>", "2"),
            ("</s>", "3"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
            ("<s>", "<s>"),
        }:
            return 10
        return 33

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_5_outputs, attn_1_1_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(mlp_0_1_output, mlp_0_3_output):
        key = (mlp_0_1_output, mlp_0_3_output)
        return 26

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(mlp_0_1_outputs, mlp_0_3_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(token, attn_0_3_output):
        key = (token, attn_0_3_output)
        return 43

    mlp_1_3_outputs = [mlp_1_3(k0, k1) for k0, k1 in zip(tokens, attn_0_3_outputs)]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_6_output, num_attn_1_5_output):
        key = (num_attn_1_6_output, num_attn_1_5_output)
        return 0

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_3_output, num_attn_1_5_output):
        key = (num_attn_1_3_output, num_attn_1_5_output)
        return 28

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_2_output):
        key = num_attn_1_2_output
        return 39

    num_mlp_1_2_outputs = [num_mlp_1_2(k0) for k0 in num_attn_1_2_outputs]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_0_6_output, num_attn_1_4_output):
        key = (num_attn_0_6_output, num_attn_1_4_output)
        return 40

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_0_6_outputs, num_attn_1_4_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(mlp_0_3_output, num_mlp_0_0_output):
        if mlp_0_3_output in {0, 26, 43}:
            return num_mlp_0_0_output == 6
        elif mlp_0_3_output in {1, 41}:
            return num_mlp_0_0_output == 20
        elif mlp_0_3_output in {2, 36, 44}:
            return num_mlp_0_0_output == 1
        elif mlp_0_3_output in {3}:
            return num_mlp_0_0_output == 44
        elif mlp_0_3_output in {4, 39}:
            return num_mlp_0_0_output == 8
        elif mlp_0_3_output in {5}:
            return num_mlp_0_0_output == 45
        elif mlp_0_3_output in {37, 6}:
            return num_mlp_0_0_output == 7
        elif mlp_0_3_output in {7}:
            return num_mlp_0_0_output == 49
        elif mlp_0_3_output in {8, 17, 21, 14}:
            return num_mlp_0_0_output == 5
        elif mlp_0_3_output in {9, 45}:
            return num_mlp_0_0_output == 14
        elif mlp_0_3_output in {10}:
            return num_mlp_0_0_output == 34
        elif mlp_0_3_output in {32, 34, 11, 13, 47, 48, 18, 19, 22, 23, 28, 29, 30, 31}:
            return num_mlp_0_0_output == 3
        elif mlp_0_3_output in {40, 12}:
            return num_mlp_0_0_output == 39
        elif mlp_0_3_output in {15}:
            return num_mlp_0_0_output == 26
        elif mlp_0_3_output in {16}:
            return num_mlp_0_0_output == 25
        elif mlp_0_3_output in {27, 20}:
            return num_mlp_0_0_output == 21
        elif mlp_0_3_output in {24}:
            return num_mlp_0_0_output == 11
        elif mlp_0_3_output in {25}:
            return num_mlp_0_0_output == 2
        elif mlp_0_3_output in {33}:
            return num_mlp_0_0_output == 35
        elif mlp_0_3_output in {35}:
            return num_mlp_0_0_output == 0
        elif mlp_0_3_output in {46, 38}:
            return num_mlp_0_0_output == 36
        elif mlp_0_3_output in {42}:
            return num_mlp_0_0_output == 13
        elif mlp_0_3_output in {49}:
            return num_mlp_0_0_output == 4

    attn_2_0_pattern = select_closest(
        num_mlp_0_0_outputs, mlp_0_3_outputs, predicate_2_0
    )
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_2_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_4_output, mlp_0_1_output):
        if attn_0_4_output in {"0"}:
            return mlp_0_1_output == 4
        elif attn_0_4_output in {"1"}:
            return mlp_0_1_output == 42
        elif attn_0_4_output in {"2"}:
            return mlp_0_1_output == 5
        elif attn_0_4_output in {"3"}:
            return mlp_0_1_output == 6
        elif attn_0_4_output in {"4"}:
            return mlp_0_1_output == 7
        elif attn_0_4_output in {"</s>"}:
            return mlp_0_1_output == 39
        elif attn_0_4_output in {"<s>"}:
            return mlp_0_1_output == 11

    attn_2_1_pattern = select_closest(mlp_0_1_outputs, attn_0_4_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_6_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(mlp_0_2_output, token):
        if mlp_0_2_output in {0, 35, 6, 7, 11, 13, 46, 15, 30}:
            return token == "</s>"
        elif mlp_0_2_output in {1, 45}:
            return token == "1"
        elif mlp_0_2_output in {
            2,
            3,
            4,
            8,
            12,
            14,
            16,
            18,
            19,
            20,
            24,
            25,
            26,
            29,
            32,
            33,
            34,
            36,
            37,
            41,
            44,
            48,
        }:
            return token == ""
        elif mlp_0_2_output in {5, 38, 47, 49, 21, 23, 27}:
            return token == "<s>"
        elif mlp_0_2_output in {39, 9, 42, 43, 22, 28}:
            return token == "4"
        elif mlp_0_2_output in {10}:
            return token == "0"
        elif mlp_0_2_output in {40, 17}:
            return token == "3"
        elif mlp_0_2_output in {31}:
            return token == "2"

    attn_2_2_pattern = select_closest(tokens, mlp_0_2_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_7_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(mlp_0_2_output, mlp_0_3_output):
        if mlp_0_2_output in {0}:
            return mlp_0_3_output == 5
        elif mlp_0_2_output in {1, 10, 12, 16, 21, 25}:
            return mlp_0_3_output == 14
        elif mlp_0_2_output in {2}:
            return mlp_0_3_output == 22
        elif mlp_0_2_output in {40, 17, 3, 7}:
            return mlp_0_3_output == 37
        elif mlp_0_2_output in {9, 34, 4}:
            return mlp_0_3_output == 47
        elif mlp_0_2_output in {8, 42, 5, 14}:
            return mlp_0_3_output == 10
        elif mlp_0_2_output in {6}:
            return mlp_0_3_output == 13
        elif mlp_0_2_output in {24, 35, 11, 28}:
            return mlp_0_3_output == 6
        elif mlp_0_2_output in {13, 22}:
            return mlp_0_3_output == 26
        elif mlp_0_2_output in {20, 15}:
            return mlp_0_3_output == 24
        elif mlp_0_2_output in {18}:
            return mlp_0_3_output == 4
        elif mlp_0_2_output in {19}:
            return mlp_0_3_output == 20
        elif mlp_0_2_output in {23}:
            return mlp_0_3_output == 28
        elif mlp_0_2_output in {26}:
            return mlp_0_3_output == 17
        elif mlp_0_2_output in {27, 47}:
            return mlp_0_3_output == 38
        elif mlp_0_2_output in {29}:
            return mlp_0_3_output == 46
        elif mlp_0_2_output in {44, 37, 30, 31}:
            return mlp_0_3_output == 2
        elif mlp_0_2_output in {32}:
            return mlp_0_3_output == 0
        elif mlp_0_2_output in {33}:
            return mlp_0_3_output == 44
        elif mlp_0_2_output in {36}:
            return mlp_0_3_output == 36
        elif mlp_0_2_output in {38}:
            return mlp_0_3_output == 31
        elif mlp_0_2_output in {39}:
            return mlp_0_3_output == 32
        elif mlp_0_2_output in {41}:
            return mlp_0_3_output == 43
        elif mlp_0_2_output in {49, 43}:
            return mlp_0_3_output == 21
        elif mlp_0_2_output in {45}:
            return mlp_0_3_output == 41
        elif mlp_0_2_output in {46}:
            return mlp_0_3_output == 35
        elif mlp_0_2_output in {48}:
            return mlp_0_3_output == 33

    attn_2_3_pattern = select_closest(mlp_0_3_outputs, mlp_0_2_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_5_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(mlp_0_2_output, attn_0_1_output):
        if mlp_0_2_output in {
            0,
            1,
            2,
            6,
            8,
            9,
            10,
            12,
            13,
            16,
            19,
            21,
            22,
            23,
            26,
            29,
            30,
            34,
            35,
            36,
            38,
            39,
            42,
            44,
            45,
            46,
            48,
            49,
        }:
            return attn_0_1_output == ""
        elif mlp_0_2_output in {32, 3, 43, 47, 25, 27}:
            return attn_0_1_output == "0"
        elif mlp_0_2_output in {4, 5, 40, 11, 15, 17}:
            return attn_0_1_output == "<s>"
        elif mlp_0_2_output in {28, 7}:
            return attn_0_1_output == "3"
        elif mlp_0_2_output in {14}:
            return attn_0_1_output == "</s>"
        elif mlp_0_2_output in {24, 18}:
            return attn_0_1_output == "1"
        elif mlp_0_2_output in {41, 20, 31}:
            return attn_0_1_output == "4"
        elif mlp_0_2_output in {33, 37}:
            return attn_0_1_output == "2"

    attn_2_4_pattern = select_closest(attn_0_1_outputs, mlp_0_2_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_5_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_2_output, attn_1_1_output):
        if attn_0_2_output in {"0", "3", "1"}:
            return attn_1_1_output == "</s>"
        elif attn_0_2_output in {"</s>", "2"}:
            return attn_1_1_output == "<s>"
        elif attn_0_2_output in {"4"}:
            return attn_1_1_output == "1"
        elif attn_0_2_output in {"<s>"}:
            return attn_1_1_output == "2"

    attn_2_5_pattern = select_closest(attn_1_1_outputs, attn_0_2_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_7_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(q_mlp_0_3_output, k_mlp_0_3_output):
        if q_mlp_0_3_output in {0, 7, 12, 15, 49, 24}:
            return k_mlp_0_3_output == 6
        elif q_mlp_0_3_output in {1, 43}:
            return k_mlp_0_3_output == 8
        elif q_mlp_0_3_output in {2}:
            return k_mlp_0_3_output == 28
        elif q_mlp_0_3_output in {3, 6}:
            return k_mlp_0_3_output == 25
        elif q_mlp_0_3_output in {4}:
            return k_mlp_0_3_output == 3
        elif q_mlp_0_3_output in {5}:
            return k_mlp_0_3_output == 0
        elif q_mlp_0_3_output in {8}:
            return k_mlp_0_3_output == 23
        elif q_mlp_0_3_output in {9, 42, 46, 39}:
            return k_mlp_0_3_output == 5
        elif q_mlp_0_3_output in {10, 18}:
            return k_mlp_0_3_output == 33
        elif q_mlp_0_3_output in {35, 11}:
            return k_mlp_0_3_output == 31
        elif q_mlp_0_3_output in {13}:
            return k_mlp_0_3_output == 20
        elif q_mlp_0_3_output in {14}:
            return k_mlp_0_3_output == 17
        elif q_mlp_0_3_output in {16, 19}:
            return k_mlp_0_3_output == 13
        elif q_mlp_0_3_output in {17}:
            return k_mlp_0_3_output == 43
        elif q_mlp_0_3_output in {20, 21}:
            return k_mlp_0_3_output == 7
        elif q_mlp_0_3_output in {22}:
            return k_mlp_0_3_output == 4
        elif q_mlp_0_3_output in {23}:
            return k_mlp_0_3_output == 24
        elif q_mlp_0_3_output in {25, 26}:
            return k_mlp_0_3_output == 21
        elif q_mlp_0_3_output in {27}:
            return k_mlp_0_3_output == 45
        elif q_mlp_0_3_output in {33, 28}:
            return k_mlp_0_3_output == 27
        elif q_mlp_0_3_output in {29}:
            return k_mlp_0_3_output == 18
        elif q_mlp_0_3_output in {30}:
            return k_mlp_0_3_output == 34
        elif q_mlp_0_3_output in {41, 47, 31}:
            return k_mlp_0_3_output == 14
        elif q_mlp_0_3_output in {32, 36}:
            return k_mlp_0_3_output == 10
        elif q_mlp_0_3_output in {34, 37}:
            return k_mlp_0_3_output == 47
        elif q_mlp_0_3_output in {44, 38}:
            return k_mlp_0_3_output == 11
        elif q_mlp_0_3_output in {40, 45}:
            return k_mlp_0_3_output == 37
        elif q_mlp_0_3_output in {48}:
            return k_mlp_0_3_output == 19

    attn_2_6_pattern = select_closest(mlp_0_3_outputs, mlp_0_3_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_1_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_0_5_output, position):
        if attn_0_5_output in {"0"}:
            return position == 10
        elif attn_0_5_output in {"1", "4"}:
            return position == 5
        elif attn_0_5_output in {"2"}:
            return position == 12
        elif attn_0_5_output in {"3"}:
            return position == 2
        elif attn_0_5_output in {"</s>"}:
            return position == 6
        elif attn_0_5_output in {"<s>"}:
            return position == 17

    attn_2_7_pattern = select_closest(positions, attn_0_5_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_0_0_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_0_3_output, position):
        if mlp_0_3_output in {0, 3, 4, 35, 38, 16, 20, 22}:
            return position == 10
        elif mlp_0_3_output in {
            1,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            15,
            21,
            26,
            27,
            28,
            30,
            32,
            37,
            40,
            41,
            42,
            43,
            45,
            47,
            48,
            49,
        }:
            return position == 2
        elif mlp_0_3_output in {2}:
            return position == 1
        elif mlp_0_3_output in {5}:
            return position == 12
        elif mlp_0_3_output in {13}:
            return position == 30
        elif mlp_0_3_output in {18, 34, 14}:
            return position == 29
        elif mlp_0_3_output in {17}:
            return position == 36
        elif mlp_0_3_output in {19}:
            return position == 4
        elif mlp_0_3_output in {23}:
            return position == 46
        elif mlp_0_3_output in {24}:
            return position == 25
        elif mlp_0_3_output in {25, 46}:
            return position == 3
        elif mlp_0_3_output in {29}:
            return position == 28
        elif mlp_0_3_output in {31}:
            return position == 38
        elif mlp_0_3_output in {33}:
            return position == 23
        elif mlp_0_3_output in {36}:
            return position == 15
        elif mlp_0_3_output in {39}:
            return position == 34
        elif mlp_0_3_output in {44}:
            return position == 5

    num_attn_2_0_pattern = select(positions, mlp_0_3_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_7_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_6_output, num_mlp_1_1_output):
        if attn_1_6_output in {"0"}:
            return num_mlp_1_1_output == 26
        elif attn_1_6_output in {"1", "2"}:
            return num_mlp_1_1_output == 47
        elif attn_1_6_output in {"3"}:
            return num_mlp_1_1_output == 8
        elif attn_1_6_output in {"4"}:
            return num_mlp_1_1_output == 15
        elif attn_1_6_output in {"</s>"}:
            return num_mlp_1_1_output == 30
        elif attn_1_6_output in {"<s>"}:
            return num_mlp_1_1_output == 40

    num_attn_2_1_pattern = select(
        num_mlp_1_1_outputs, attn_1_6_outputs, num_predicate_2_1
    )
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_7_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(mlp_0_3_output, attn_0_3_output):
        if mlp_0_3_output in {0, 32, 5, 6, 38, 8, 41, 11, 43, 45, 14, 15, 48, 17, 28}:
            return attn_0_3_output == ""
        elif mlp_0_3_output in {40, 1, 9, 47}:
            return attn_0_3_output == "<s>"
        elif mlp_0_3_output in {2, 10, 12, 49, 26, 27, 30}:
            return attn_0_3_output == "</s>"
        elif mlp_0_3_output in {37, 3, 20, 21}:
            return attn_0_3_output == "1"
        elif mlp_0_3_output in {35, 4}:
            return attn_0_3_output == "2"
        elif mlp_0_3_output in {
            33,
            34,
            36,
            7,
            39,
            42,
            44,
            13,
            46,
            16,
            18,
            19,
            22,
            23,
            24,
            25,
            29,
            31,
        }:
            return attn_0_3_output == "0"

    num_attn_2_2_pattern = select(attn_0_3_outputs, mlp_0_3_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_7_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(mlp_0_3_output, attn_0_6_output):
        if mlp_0_3_output in {0, 1, 32, 5, 6, 37, 38, 42}:
            return attn_0_6_output == "0"
        elif mlp_0_3_output in {
            2,
            3,
            4,
            7,
            9,
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
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            33,
            35,
            36,
            39,
            40,
            41,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
        }:
            return attn_0_6_output == ""
        elif mlp_0_3_output in {8, 24, 10, 21}:
            return attn_0_6_output == "<s>"
        elif mlp_0_3_output in {34}:
            return attn_0_6_output == "</s>"

    num_attn_2_3_pattern = select(attn_0_6_outputs, mlp_0_3_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_0_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(position, attn_0_5_output):
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
            12,
            13,
            14,
            16,
            39,
            41,
            42,
            44,
            45,
            49,
        }:
            return attn_0_5_output == "1"
        elif position in {1, 17, 18, 21, 22, 23, 24}:
            return attn_0_5_output == "<s>"
        elif position in {43, 15}:
            return attn_0_5_output == "0"
        elif position in {25, 19}:
            return attn_0_5_output == "</s>"
        elif position in {20}:
            return attn_0_5_output == "2"
        elif position in {
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            40,
            46,
            47,
            48,
            26,
            27,
            28,
            29,
            30,
            31,
        }:
            return attn_0_5_output == ""

    num_attn_2_4_pattern = select(attn_0_5_outputs, positions, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_1_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_1_6_output, attn_1_4_output):
        if attn_1_6_output in {"</s>", "0", "1", "4"}:
            return attn_1_4_output == ""
        elif attn_1_6_output in {"2"}:
            return attn_1_4_output == "2"
        elif attn_1_6_output in {"3"}:
            return attn_1_4_output == "</s>"
        elif attn_1_6_output in {"<s>"}:
            return attn_1_4_output == "<pad>"

    num_attn_2_5_pattern = select(attn_1_4_outputs, attn_1_6_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_1_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_6_output, attn_1_4_output):
        if attn_1_6_output in {"0"}:
            return attn_1_4_output == "1"
        elif attn_1_6_output in {"1", "2"}:
            return attn_1_4_output == "<s>"
        elif attn_1_6_output in {"3", "4"}:
            return attn_1_4_output == ""
        elif attn_1_6_output in {"</s>", "<s>"}:
            return attn_1_4_output == "0"

    num_attn_2_6_pattern = select(attn_1_4_outputs, attn_1_6_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_2_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(mlp_1_2_output, token):
        if mlp_1_2_output in {0, 2, 39, 44, 46, 17, 18, 19, 21, 23, 24, 25, 26}:
            return token == "<s>"
        elif mlp_1_2_output in {48, 1, 4}:
            return token == "</s>"
        elif mlp_1_2_output in {3, 6, 7, 8, 9, 42, 43, 12, 13, 14, 49, 20}:
            return token == "1"
        elif mlp_1_2_output in {32, 35, 36, 5, 37, 38, 41, 47, 27, 29, 31}:
            return token == ""
        elif mlp_1_2_output in {10, 28, 22}:
            return token == "2"
        elif mlp_1_2_output in {40, 11, 45, 15, 16}:
            return token == "0"
        elif mlp_1_2_output in {33, 34, 30}:
            return token == "<pad>"

    num_attn_2_7_pattern = select(tokens, mlp_1_2_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_7_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(mlp_0_2_output, attn_2_3_output):
        key = (mlp_0_2_output, attn_2_3_output)
        if key in {
            (1, "0"),
            (3, "0"),
            (3, "1"),
            (6, "0"),
            (6, "1"),
            (6, "2"),
            (6, "3"),
            (6, "4"),
            (6, "</s>"),
            (6, "<s>"),
            (10, "0"),
            (10, "1"),
            (10, "2"),
            (10, "3"),
            (10, "4"),
            (10, "</s>"),
            (10, "<s>"),
            (14, "0"),
            (14, "1"),
            (14, "2"),
            (14, "3"),
            (14, "4"),
            (14, "</s>"),
            (14, "<s>"),
            (16, "0"),
            (16, "1"),
            (16, "2"),
            (16, "3"),
            (16, "4"),
            (16, "</s>"),
            (16, "<s>"),
            (19, "0"),
            (26, "0"),
            (26, "1"),
            (26, "2"),
            (26, "3"),
            (26, "4"),
            (26, "</s>"),
            (26, "<s>"),
            (45, "0"),
            (45, "1"),
        }:
            return 20
        return 13

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(mlp_0_2_outputs, attn_2_3_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(position, mlp_0_2_output):
        key = (position, mlp_0_2_output)
        if key in {
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (1, 13),
            (1, 14),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (1, 20),
            (1, 21),
            (1, 22),
            (1, 23),
            (1, 24),
            (1, 25),
            (1, 26),
            (1, 27),
            (1, 28),
            (1, 29),
            (1, 30),
            (1, 31),
            (1, 32),
            (1, 33),
            (1, 34),
            (1, 35),
            (1, 36),
            (1, 37),
            (1, 38),
            (1, 39),
            (1, 40),
            (1, 41),
            (1, 42),
            (1, 43),
            (1, 44),
            (1, 45),
            (1, 46),
            (1, 47),
            (1, 48),
            (1, 49),
        }:
            return 9
        return 0

    mlp_2_1_outputs = [mlp_2_1(k0, k1) for k0, k1 in zip(positions, mlp_0_2_outputs)]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(position, attn_0_5_output):
        key = (position, attn_0_5_output)
        if key in {
            (0, "1"),
            (0, "3"),
            (0, "4"),
            (0, "<s>"),
            (1, "4"),
            (2, "4"),
            (3, "4"),
            (3, "<s>"),
            (5, "4"),
            (5, "<s>"),
            (6, "4"),
            (8, "0"),
            (8, "1"),
            (8, "2"),
            (8, "3"),
            (8, "4"),
            (8, "</s>"),
            (8, "<s>"),
            (9, "1"),
            (9, "3"),
            (9, "4"),
            (9, "<s>"),
            (10, "4"),
            (11, "1"),
            (11, "3"),
            (11, "4"),
            (11, "<s>"),
            (12, "4"),
            (12, "<s>"),
            (13, "1"),
            (13, "3"),
            (13, "4"),
            (13, "<s>"),
            (17, "1"),
            (17, "3"),
            (17, "4"),
            (17, "<s>"),
            (18, "4"),
            (21, "4"),
            (22, "4"),
            (23, "4"),
            (25, "3"),
            (25, "4"),
            (25, "<s>"),
            (26, "1"),
            (26, "2"),
            (26, "3"),
            (26, "4"),
            (26, "<s>"),
            (27, "1"),
            (27, "3"),
            (27, "4"),
            (27, "<s>"),
            (29, "1"),
            (29, "3"),
            (29, "4"),
            (29, "<s>"),
            (30, "4"),
            (31, "1"),
            (31, "3"),
            (31, "4"),
            (31, "<s>"),
            (32, "0"),
            (32, "1"),
            (32, "2"),
            (32, "3"),
            (32, "4"),
            (32, "<s>"),
            (33, "1"),
            (33, "2"),
            (33, "3"),
            (33, "4"),
            (33, "<s>"),
            (34, "4"),
            (35, "4"),
            (35, "<s>"),
            (36, "1"),
            (36, "3"),
            (36, "4"),
            (36, "<s>"),
            (37, "4"),
            (38, "4"),
            (38, "<s>"),
            (39, "1"),
            (39, "3"),
            (39, "4"),
            (39, "<s>"),
            (40, "1"),
            (40, "3"),
            (40, "4"),
            (40, "<s>"),
            (41, "1"),
            (41, "3"),
            (41, "4"),
            (41, "<s>"),
            (42, "1"),
            (42, "3"),
            (42, "4"),
            (42, "<s>"),
            (43, "4"),
            (44, "1"),
            (44, "3"),
            (44, "4"),
            (44, "<s>"),
            (45, "1"),
            (45, "3"),
            (45, "4"),
            (45, "<s>"),
            (46, "3"),
            (46, "4"),
            (46, "<s>"),
            (47, "1"),
            (47, "3"),
            (47, "4"),
            (47, "<s>"),
            (48, "1"),
            (48, "3"),
            (48, "4"),
            (48, "<s>"),
            (49, "4"),
        }:
            return 43
        return 4

    mlp_2_2_outputs = [mlp_2_2(k0, k1) for k0, k1 in zip(positions, attn_0_5_outputs)]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(attn_2_1_output, num_mlp_1_2_output):
        key = (attn_2_1_output, num_mlp_1_2_output)
        return 28

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(attn_2_1_outputs, num_mlp_1_2_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_4_output, num_attn_1_6_output):
        key = (num_attn_1_4_output, num_attn_1_6_output)
        return 19

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_6_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_6_output, num_attn_0_5_output):
        key = (num_attn_2_6_output, num_attn_0_5_output)
        return 11

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_6_outputs, num_attn_0_5_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_2_3_output, num_attn_2_2_output):
        key = (num_attn_2_3_output, num_attn_2_2_output)
        return 27

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_2_2_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_2_1_output, num_attn_2_0_output):
        key = (num_attn_2_1_output, num_attn_2_0_output)
        return 1

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_2_0_outputs)
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
            "3",
            "4",
            "0",
            "1",
            "3",
            "0",
            "0",
            "1",
            "4",
            "4",
            "1",
            "2",
            "4",
            "2",
            "4",
            "3",
            "4",
            "2",
            "4",
            "2",
            "4",
            "1",
            "1",
            "0",
            "1",
            "1",
            "1",
            "1",
            "0",
            "4",
            "1",
            "0",
            "0",
            "3",
            "2",
            "1",
            "0",
            "3",
            "</s>",
        ]
    )
)
