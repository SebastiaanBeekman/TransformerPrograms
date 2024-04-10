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
        "output/length/rasp/sort/trainlength30/s0/sort_weights.csv",
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
        if q_position in {0, 27}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2, 28, 29}:
            return k_position == 4
        elif q_position in {8, 17, 3}:
            return k_position == 5
        elif q_position in {4, 12}:
            return k_position == 8
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {24, 6}:
            return k_position == 9
        elif q_position in {16, 7}:
            return k_position == 2
        elif q_position in {9}:
            return k_position == 14
        elif q_position in {10}:
            return k_position == 13
        elif q_position in {11, 15}:
            return k_position == 16
        elif q_position in {21, 13}:
            return k_position == 11
        elif q_position in {14}:
            return k_position == 17
        elif q_position in {18, 19}:
            return k_position == 21
        elif q_position in {20}:
            return k_position == 22
        elif q_position in {22}:
            return k_position == 23
        elif q_position in {23}:
            return k_position == 24
        elif q_position in {25}:
            return k_position == 26
        elif q_position in {26}:
            return k_position == 27
        elif q_position in {30}:
            return k_position == 19
        elif q_position in {31}:
            return k_position == 34
        elif q_position in {32}:
            return k_position == 15
        elif q_position in {33, 37, 38}:
            return k_position == 36
        elif q_position in {34}:
            return k_position == 30
        elif q_position in {35}:
            return k_position == 37
        elif q_position in {36, 39}:
            return k_position == 0

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2", "</s>", "<s>", "4"}:
            return k_token == "4"
        elif q_token in {"3"}:
            return k_token == "3"

    attn_0_1_pattern = select_closest(tokens, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 34}:
            return k_position == 1
        elif q_position in {1, 9}:
            return k_position == 2
        elif q_position in {2, 29}:
            return k_position == 3
        elif q_position in {25, 3, 28, 13}:
            return k_position == 4
        elif q_position in {4, 14}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 9
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8, 37}:
            return k_position == 13
        elif q_position in {24, 10, 27}:
            return k_position == 6
        elif q_position in {11}:
            return k_position == 17
        elif q_position in {12}:
            return k_position == 15
        elif q_position in {15}:
            return k_position == 21
        elif q_position in {16}:
            return k_position == 20
        elif q_position in {17}:
            return k_position == 12
        elif q_position in {18, 20}:
            return k_position == 24
        elif q_position in {19}:
            return k_position == 23
        elif q_position in {21, 23}:
            return k_position == 25
        elif q_position in {26, 22}:
            return k_position == 27
        elif q_position in {38, 30, 31}:
            return k_position == 31
        elif q_position in {32}:
            return k_position == 36
        elif q_position in {33}:
            return k_position == 34
        elif q_position in {35, 36, 39}:
            return k_position == 37

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0, 25, 28}:
            return token == "<s>"
        elif position in {1, 21}:
            return token == "0"
        elif position in {2, 3, 8, 9, 13, 19, 24}:
            return token == "1"
        elif position in {4, 5, 10, 14, 18, 20, 22, 26, 27, 29}:
            return token == "3"
        elif position in {6, 7, 11, 12, 15, 23}:
            return token == "4"
        elif position in {16, 17}:
            return token == "2"
        elif position in {32, 33, 34, 35, 36, 37, 38, 39, 30}:
            return token == ""
        elif position in {31}:
            return token == "</s>"

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(position, token):
        if position in {0, 4, 6, 8, 10, 11, 12, 14, 16, 17, 18, 20, 23, 26, 28, 29}:
            return token == "3"
        elif position in {1, 2, 7, 13, 15}:
            return token == "1"
        elif position in {3}:
            return token == "2"
        elif position in {9, 21, 5}:
            return token == "4"
        elif position in {32, 33, 34, 35, 36, 37, 38, 39, 19, 27, 30, 31}:
            return token == ""
        elif position in {25, 22}:
            return token == "<s>"
        elif position in {24}:
            return token == "0"

    attn_0_4_pattern = select_closest(tokens, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(position, token):
        if position in {0, 1, 2, 10, 13, 14, 15, 16, 18, 19, 20, 22, 26}:
            return token == "4"
        elif position in {3, 4, 5, 6, 7, 9, 11, 21, 27}:
            return token == "3"
        elif position in {8, 17, 28}:
            return token == "1"
        elif position in {35, 36, 37, 39, 12}:
            return token == ""
        elif position in {33, 34, 38, 23, 29, 30}:
            return token == "2"
        elif position in {24, 31}:
            return token == "<s>"
        elif position in {25}:
            return token == "</s>"
        elif position in {32}:
            return token == "0"

    attn_0_5_pattern = select_closest(tokens, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 4}:
            return k_position == 6
        elif q_position in {1, 15}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {27, 5, 6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 11
        elif q_position in {11, 13}:
            return k_position == 14
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {14}:
            return k_position == 15
        elif q_position in {16}:
            return k_position == 17
        elif q_position in {17}:
            return k_position == 18
        elif q_position in {18}:
            return k_position == 19
        elif q_position in {19}:
            return k_position == 20
        elif q_position in {20, 30}:
            return k_position == 21
        elif q_position in {21}:
            return k_position == 22
        elif q_position in {37, 22, 39}:
            return k_position == 24
        elif q_position in {24, 23}:
            return k_position == 25
        elif q_position in {25}:
            return k_position == 27
        elif q_position in {26}:
            return k_position == 28
        elif q_position in {28, 38}:
            return k_position == 29
        elif q_position in {32, 35, 29}:
            return k_position == 3
        elif q_position in {33, 31}:
            return k_position == 34
        elif q_position in {34}:
            return k_position == 39
        elif q_position in {36}:
            return k_position == 30

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1", "<s>"}:
            return k_token == "1"
        elif q_token in {"2", "</s>"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"

    attn_0_7_pattern = select_closest(tokens, tokens, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 10
        elif q_position in {1}:
            return k_position == 19
        elif q_position in {2, 4, 29}:
            return k_position == 6
        elif q_position in {3}:
            return k_position == 7
        elif q_position in {5, 6}:
            return k_position == 9
        elif q_position in {9, 7}:
            return k_position == 12
        elif q_position in {8}:
            return k_position == 13
        elif q_position in {10, 12}:
            return k_position == 15
        elif q_position in {11}:
            return k_position == 14
        elif q_position in {13}:
            return k_position == 16
        elif q_position in {14}:
            return k_position == 17
        elif q_position in {15}:
            return k_position == 18
        elif q_position in {16, 34, 19}:
            return k_position == 23
        elif q_position in {17}:
            return k_position == 20
        elif q_position in {18, 20}:
            return k_position == 24
        elif q_position in {33, 35, 21, 38}:
            return k_position == 25
        elif q_position in {22, 31}:
            return k_position == 26
        elif q_position in {24, 23}:
            return k_position == 27
        elif q_position in {25}:
            return k_position == 29
        elif q_position in {26}:
            return k_position == 33
        elif q_position in {27}:
            return k_position == 35
        elif q_position in {28}:
            return k_position == 39
        elif q_position in {32, 30}:
            return k_position == 21
        elif q_position in {36, 37}:
            return k_position == 4
        elif q_position in {39}:
            return k_position == 1

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0, 8, 6}:
            return k_position == 11
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 8
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 10
        elif q_position in {9}:
            return k_position == 14
        elif q_position in {10}:
            return k_position == 13
        elif q_position in {11}:
            return k_position == 16
        elif q_position in {12, 13}:
            return k_position == 18
        elif q_position in {14}:
            return k_position == 19
        elif q_position in {33, 18, 15}:
            return k_position == 21
        elif q_position in {16, 30}:
            return k_position == 20
        elif q_position in {17, 37}:
            return k_position == 22
        elif q_position in {19}:
            return k_position == 25
        elif q_position in {20, 39}:
            return k_position == 26
        elif q_position in {21, 22}:
            return k_position == 27
        elif q_position in {24, 23}:
            return k_position == 28
        elif q_position in {25, 26, 27}:
            return k_position == 31
        elif q_position in {28}:
            return k_position == 35
        elif q_position in {32, 36, 29}:
            return k_position == 23
        elif q_position in {31}:
            return k_position == 29
        elif q_position in {34, 38}:
            return k_position == 24
        elif q_position in {35}:
            return k_position == 17

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 28}:
            return token == "0"
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
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            19,
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
        elif position in {18, 20}:
            return token == "<s>"
        elif position in {21}:
            return token == "</s>"
        elif position in {22, 23, 24, 25, 26}:
            return token == "4"
        elif position in {27, 29}:
            return token == "1"

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
            return token == "</s>"
        elif position in {32, 2, 3, 4, 5, 6, 34, 36, 39}:
            return token == "0"
        elif position in {
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
            33,
            35,
            37,
            38,
        }:
            return token == ""

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {0, 1, 2, 33, 34, 5, 35, 36, 37, 38, 39, 26, 28, 29, 30, 31}:
            return token == ""
        elif position in {3, 4}:
            return token == "<pad>"
        elif position in {27, 6, 7}:
            return token == "<s>"
        elif position in {32, 8, 9, 11, 13, 22}:
            return token == "1"
        elif position in {10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25}:
            return token == "0"
        elif position in {24}:
            return token == "</s>"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {0, 25, 28}:
            return token == "0"
        elif position in {32, 1, 2, 3, 4, 5, 33, 34, 35, 36, 37, 38, 39, 29, 30, 31}:
            return token == ""
        elif position in {10, 15, 6, 7}:
            return token == "<s>"
        elif position in {8, 9, 11, 12, 13, 14}:
            return token == "</s>"
        elif position in {16, 17}:
            return token == "2"
        elif position in {18, 19, 20, 21, 22, 23, 24, 26, 27}:
            return token == "1"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {
            0,
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
        }:
            return token == ""
        elif position in {1}:
            return token == "</s>"
        elif position in {32, 2, 3, 4, 5, 6, 34, 35, 36, 10, 37, 38, 39, 30}:
            return token == "1"
        elif position in {7, 8, 9, 11, 29}:
            return token == "0"
        elif position in {31}:
            return token == "2"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {0, 26, 27}:
            return token == "2"
        elif position in {1, 2}:
            return token == "0"
        elif position in {
            32,
            33,
            34,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            35,
            11,
            36,
            37,
            38,
            39,
            30,
            31,
        }:
            return token == ""
        elif position in {10, 12, 15}:
            return token == "<s>"
        elif position in {13, 14, 16, 17, 18}:
            return token == "</s>"
        elif position in {19, 20, 21}:
            return token == "4"
        elif position in {22, 23, 24, 25, 28, 29}:
            return token == "1"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_4_output, position):
        key = (attn_0_4_output, position)
        if key in {
            ("0", 1),
            ("0", 25),
            ("0", 26),
            ("0", 27),
            ("1", 1),
            ("1", 25),
            ("1", 26),
            ("1", 27),
            ("2", 1),
            ("2", 25),
            ("2", 26),
            ("2", 27),
            ("3", 1),
            ("3", 25),
            ("3", 26),
            ("3", 27),
            ("4", 1),
            ("</s>", 1),
            ("</s>", 25),
            ("</s>", 26),
            ("</s>", 27),
            ("<s>", 1),
            ("<s>", 25),
            ("<s>", 26),
            ("<s>", 27),
        }:
            return 26
        elif key in {
            ("0", 15),
            ("1", 15),
            ("2", 15),
            ("3", 15),
            ("4", 15),
            ("</s>", 15),
            ("<s>", 15),
        }:
            return 17
        return 6

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_4_outputs, positions)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position):
        key = position
        if key in {5, 32, 33, 34, 35}:
            return 6
        elif key in {1, 2, 3}:
            return 2
        elif key in {0, 6, 8}:
            return 7
        elif key in {4}:
            return 38
        return 11

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in positions]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_6_output, position):
        key = (attn_0_6_output, position)
        if key in {
            ("0", 3),
            ("0", 4),
            ("0", 9),
            ("0", 10),
            ("0", 11),
            ("0", 12),
            ("0", 30),
            ("0", 31),
            ("0", 37),
            ("0", 39),
            ("1", 3),
            ("1", 4),
            ("1", 9),
            ("1", 10),
            ("1", 11),
            ("1", 12),
            ("1", 30),
            ("1", 37),
            ("2", 3),
            ("2", 4),
            ("2", 9),
            ("2", 10),
            ("2", 11),
            ("2", 12),
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
            ("3", 3),
            ("3", 4),
            ("3", 9),
            ("3", 10),
            ("3", 11),
            ("3", 12),
            ("3", 30),
            ("3", 31),
            ("3", 32),
            ("3", 33),
            ("3", 34),
            ("3", 35),
            ("3", 36),
            ("3", 37),
            ("3", 38),
            ("3", 39),
            ("4", 3),
            ("4", 4),
            ("4", 9),
            ("4", 10),
            ("4", 11),
            ("4", 12),
            ("4", 30),
            ("4", 31),
            ("4", 32),
            ("4", 33),
            ("4", 34),
            ("4", 35),
            ("4", 36),
            ("4", 37),
            ("4", 38),
            ("4", 39),
            ("</s>", 3),
            ("</s>", 4),
            ("</s>", 9),
            ("</s>", 10),
            ("</s>", 11),
            ("</s>", 12),
            ("</s>", 30),
            ("</s>", 31),
            ("</s>", 37),
            ("</s>", 39),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 9),
            ("<s>", 10),
            ("<s>", 11),
            ("<s>", 12),
            ("<s>", 30),
            ("<s>", 37),
            ("<s>", 39),
        }:
            return 25
        elif key in {
            ("0", 1),
            ("0", 2),
            ("1", 1),
            ("1", 2),
            ("2", 1),
            ("2", 2),
            ("3", 1),
            ("3", 2),
            ("4", 1),
            ("4", 2),
            ("</s>", 1),
            ("</s>", 2),
            ("<s>", 1),
            ("<s>", 2),
        }:
            return 6
        return 26

    mlp_0_2_outputs = [mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_6_outputs, positions)]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_5_output, position):
        key = (attn_0_5_output, position)
        if key in {
            ("0", 8),
            ("0", 9),
            ("0", 10),
            ("0", 11),
            ("0", 27),
            ("0", 28),
            ("0", 29),
            ("1", 8),
            ("1", 9),
            ("1", 10),
            ("1", 11),
            ("1", 27),
            ("1", 28),
            ("1", 29),
            ("2", 8),
            ("2", 9),
            ("2", 10),
            ("2", 11),
            ("2", 27),
            ("2", 28),
            ("2", 29),
            ("3", 8),
            ("3", 9),
            ("3", 10),
            ("3", 11),
            ("3", 27),
            ("3", 29),
            ("4", 8),
            ("4", 9),
            ("4", 10),
            ("4", 11),
            ("4", 27),
            ("4", 28),
            ("4", 29),
            ("</s>", 8),
            ("</s>", 9),
            ("</s>", 10),
            ("</s>", 11),
            ("</s>", 27),
            ("</s>", 28),
            ("</s>", 29),
            ("<s>", 8),
            ("<s>", 9),
            ("<s>", 10),
            ("<s>", 11),
            ("<s>", 27),
            ("<s>", 28),
            ("<s>", 29),
        }:
            return 14
        elif key in {
            ("0", 2),
            ("0", 3),
            ("0", 4),
            ("0", 5),
            ("0", 6),
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
            ("1", 2),
            ("1", 3),
            ("1", 4),
            ("1", 5),
            ("1", 6),
            ("1", 30),
            ("1", 32),
            ("1", 33),
            ("2", 2),
            ("2", 3),
            ("2", 4),
            ("2", 5),
            ("2", 6),
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
            ("4", 30),
            ("4", 31),
            ("4", 32),
            ("4", 37),
            ("4", 38),
            ("4", 39),
            ("</s>", 2),
            ("</s>", 3),
            ("</s>", 4),
            ("</s>", 5),
            ("<s>", 2),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 5),
        }:
            return 19
        elif key in {
            ("0", 0),
            ("0", 7),
            ("1", 0),
            ("1", 7),
            ("2", 0),
            ("2", 7),
            ("3", 0),
            ("3", 6),
            ("3", 7),
            ("3", 28),
            ("3", 30),
            ("3", 31),
            ("3", 32),
            ("3", 33),
            ("3", 34),
            ("3", 35),
            ("3", 36),
            ("3", 37),
            ("3", 38),
            ("3", 39),
            ("4", 0),
            ("4", 6),
            ("4", 7),
            ("4", 33),
            ("4", 34),
            ("4", 35),
            ("4", 36),
            ("</s>", 0),
            ("</s>", 6),
            ("</s>", 7),
            ("</s>", 30),
            ("</s>", 31),
            ("</s>", 32),
            ("</s>", 33),
            ("</s>", 34),
            ("</s>", 35),
            ("</s>", 36),
            ("</s>", 37),
            ("</s>", 38),
            ("</s>", 39),
            ("<s>", 0),
            ("<s>", 6),
            ("<s>", 7),
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
            return 38
        elif key in {("1", 34)}:
            return 11
        return 15

    mlp_0_3_outputs = [mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_5_outputs, positions)]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output, num_attn_0_0_output):
        key = (num_attn_0_2_output, num_attn_0_0_output)
        return 6

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_3_output):
        key = num_attn_0_3_output
        return 23

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_3_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_2_output, num_attn_0_7_output):
        key = (num_attn_0_2_output, num_attn_0_7_output)
        return 9

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_7_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_1_output, num_attn_0_0_output):
        key = (num_attn_0_1_output, num_attn_0_0_output)
        return 9

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 12
        elif q_position in {8, 1, 3, 7}:
            return k_position == 2
        elif q_position in {2, 38}:
            return k_position == 26
        elif q_position in {18, 4, 20}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 38
        elif q_position in {36, 6}:
            return k_position == 10
        elif q_position in {24, 9}:
            return k_position == 13
        elif q_position in {32, 10, 15, 25, 26}:
            return k_position == 5
        elif q_position in {11, 29}:
            return k_position == 4
        elif q_position in {35, 27, 12}:
            return k_position == 6
        elif q_position in {33, 19, 13}:
            return k_position == 11
        elif q_position in {14}:
            return k_position == 19
        elif q_position in {16, 17, 30, 23}:
            return k_position == 7
        elif q_position in {28, 21}:
            return k_position == 8
        elif q_position in {22}:
            return k_position == 14
        elif q_position in {31}:
            return k_position == 27
        elif q_position in {34}:
            return k_position == 34
        elif q_position in {37}:
            return k_position == 17
        elif q_position in {39}:
            return k_position == 20

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_2_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(mlp_0_0_output, token):
        if mlp_0_0_output in {
            0,
            32,
            2,
            35,
            4,
            36,
            39,
            11,
            12,
            13,
            14,
            15,
            18,
            21,
            22,
            23,
            25,
            31,
        }:
            return token == ""
        elif mlp_0_0_output in {1, 10, 16, 17, 19, 24, 27, 28, 29}:
            return token == "3"
        elif mlp_0_0_output in {8, 9, 34, 3}:
            return token == "1"
        elif mlp_0_0_output in {5}:
            return token == "2"
        elif mlp_0_0_output in {33, 37, 6, 7, 30}:
            return token == "4"
        elif mlp_0_0_output in {20}:
            return token == "<pad>"
        elif mlp_0_0_output in {26}:
            return token == "<s>"
        elif mlp_0_0_output in {38}:
            return token == "0"

    attn_1_1_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_7_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(mlp_0_1_output, attn_0_0_output):
        if mlp_0_1_output in {0, 33, 32, 3, 37, 39}:
            return attn_0_0_output == "</s>"
        elif mlp_0_1_output in {1, 12, 28}:
            return attn_0_0_output == "1"
        elif mlp_0_1_output in {2, 34, 16, 23, 24, 31}:
            return attn_0_0_output == "<s>"
        elif mlp_0_1_output in {4, 5, 9, 13, 14, 18, 19, 20, 21, 22, 26, 27}:
            return attn_0_0_output == ""
        elif mlp_0_1_output in {35, 6}:
            return attn_0_0_output == "2"
        elif mlp_0_1_output in {36, 38, 7, 17, 30}:
            return attn_0_0_output == "4"
        elif mlp_0_1_output in {8, 25, 29}:
            return attn_0_0_output == "0"
        elif mlp_0_1_output in {10, 11, 15}:
            return attn_0_0_output == "3"

    attn_1_2_pattern = select_closest(attn_0_0_outputs, mlp_0_1_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_7_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(position, mlp_0_2_output):
        if position in {0, 7}:
            return mlp_0_2_output == 37
        elif position in {1, 36}:
            return mlp_0_2_output == 1
        elif position in {32, 33, 2, 3, 4, 5, 34, 37, 39, 18, 19, 23, 27, 30, 31}:
            return mlp_0_2_output == 6
        elif position in {16, 13, 6}:
            return mlp_0_2_output == 10
        elif position in {8}:
            return mlp_0_2_output == 25
        elif position in {9, 11, 20, 21, 22}:
            return mlp_0_2_output == 26
        elif position in {10}:
            return mlp_0_2_output == 16
        elif position in {12}:
            return mlp_0_2_output == 3
        elif position in {17, 14}:
            return mlp_0_2_output == 9
        elif position in {15}:
            return mlp_0_2_output == 29
        elif position in {24, 25, 26}:
            return mlp_0_2_output == 7
        elif position in {28}:
            return mlp_0_2_output == 0
        elif position in {29}:
            return mlp_0_2_output == 4
        elif position in {35}:
            return mlp_0_2_output == 18
        elif position in {38}:
            return mlp_0_2_output == 11

    attn_1_3_pattern = select_closest(mlp_0_2_outputs, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(position, token):
        if position in {0, 32, 4, 11, 15, 18, 25}:
            return token == "4"
        elif position in {1, 34, 3, 36, 39, 8, 9, 13}:
            return token == "1"
        elif position in {2, 37, 38, 14, 19, 26, 31}:
            return token == ""
        elif position in {5, 6, 7, 17, 20, 23, 27, 28, 29}:
            return token == "3"
        elif position in {33, 10, 12, 16, 30}:
            return token == "0"
        elif position in {21}:
            return token == "<s>"
        elif position in {22}:
            return token == "</s>"
        elif position in {24, 35}:
            return token == "2"

    attn_1_4_pattern = select_closest(tokens, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, tokens)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {1, 37}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 39
        elif q_position in {3}:
            return k_position == 37
        elif q_position in {4}:
            return k_position == 29
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {36, 6, 14, 18, 26}:
            return k_position == 1
        elif q_position in {7}:
            return k_position == 4
        elif q_position in {32, 8, 10, 19, 29, 30, 31}:
            return k_position == 6
        elif q_position in {9, 23}:
            return k_position == 14
        elif q_position in {11}:
            return k_position == 16
        elif q_position in {12, 13}:
            return k_position == 15
        elif q_position in {17, 15}:
            return k_position == 18
        elif q_position in {16}:
            return k_position == 20
        elif q_position in {20}:
            return k_position == 25
        elif q_position in {21}:
            return k_position == 19
        elif q_position in {27, 22}:
            return k_position == 11
        elif q_position in {24, 39}:
            return k_position == 9
        elif q_position in {25}:
            return k_position == 13
        elif q_position in {33, 28}:
            return k_position == 26
        elif q_position in {34, 38}:
            return k_position == 8
        elif q_position in {35}:
            return k_position == 12

    attn_1_5_pattern = select_closest(positions, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_2_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(q_token, k_token):
        if q_token in {"</s>", "0"}:
            return k_token == "4"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2", "4", "3"}:
            return k_token == ""
        elif q_token in {"<s>"}:
            return k_token == "3"

    attn_1_6_pattern = select_closest(tokens, tokens, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_6_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(position, token):
        if position in {0, 8, 9, 10, 11, 12, 13, 14, 17, 23, 25}:
            return token == "4"
        elif position in {1, 3, 4, 39, 18, 20}:
            return token == "1"
        elif position in {2, 28, 7}:
            return token == "0"
        elif position in {27, 19, 5, 22}:
            return token == "</s>"
        elif position in {33, 37, 6, 15, 21, 26, 29}:
            return token == "2"
        elif position in {16, 35, 36, 38}:
            return token == ""
        elif position in {32, 34, 24, 30, 31}:
            return token == "<s>"

    attn_1_7_pattern = select_closest(tokens, positions, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, tokens)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(mlp_0_0_output, attn_0_5_output):
        if mlp_0_0_output in {
            0,
            2,
            3,
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
            15,
            18,
            21,
            24,
            26,
            30,
            31,
            32,
            33,
            34,
            36,
        }:
            return attn_0_5_output == "2"
        elif mlp_0_0_output in {1, 35, 5, 37, 38, 39, 17, 19, 20, 29}:
            return attn_0_5_output == "<s>"
        elif mlp_0_0_output in {16, 22, 23}:
            return attn_0_5_output == "</s>"
        elif mlp_0_0_output in {25, 27, 28}:
            return attn_0_5_output == ""

    num_attn_1_0_pattern = select(attn_0_5_outputs, mlp_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_7_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, token):
        if position in {
            0,
            9,
            10,
            11,
            12,
            13,
            14,
            16,
            17,
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
        elif position in {1}:
            return token == "</s>"
        elif position in {2}:
            return token == "2"
        elif position in {32, 33, 34, 3, 4, 5, 6, 7, 35, 36, 38, 39, 29, 30, 31}:
            return token == "0"
        elif position in {8, 37}:
            return token == "1"
        elif position in {18, 15}:
            return token == "<pad>"

    num_attn_1_1_pattern = select(tokens, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_4_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(num_mlp_0_0_output, token):
        if num_mlp_0_0_output in {
            0,
            32,
            33,
            3,
            4,
            5,
            36,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            20,
            29,
            30,
            31,
        }:
            return token == "2"
        elif num_mlp_0_0_output in {
            1,
            35,
            6,
            7,
            38,
            9,
            39,
            17,
            18,
            19,
            21,
            22,
            23,
            24,
            26,
        }:
            return token == "<s>"
        elif num_mlp_0_0_output in {25, 2, 37}:
            return token == ""
        elif num_mlp_0_0_output in {8, 34, 27, 28}:
            return token == "</s>"

    num_attn_1_2_pattern = select(tokens, num_mlp_0_0_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_7_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(mlp_0_3_output, attn_0_3_output):
        if mlp_0_3_output in {
            0,
            3,
            4,
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
            17,
            18,
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
            31,
            32,
            33,
            35,
            36,
            37,
            39,
        }:
            return attn_0_3_output == ""
        elif mlp_0_3_output in {1, 2, 6}:
            return attn_0_3_output == "<s>"
        elif mlp_0_3_output in {19, 38}:
            return attn_0_3_output == "1"
        elif mlp_0_3_output in {34, 23}:
            return attn_0_3_output == "<pad>"

    num_attn_1_3_pattern = select(attn_0_3_outputs, mlp_0_3_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_4_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(position, attn_0_3_output):
        if position in {0, 32, 2, 3, 33, 34, 35, 36, 8, 37, 38, 11, 12, 15, 18, 19, 30}:
            return attn_0_3_output == "0"
        elif position in {1}:
            return attn_0_3_output == "2"
        elif position in {4, 5, 6, 39, 21, 24, 25, 26, 27}:
            return attn_0_3_output == ""
        elif position in {29, 13, 22, 7}:
            return attn_0_3_output == "<s>"
        elif position in {16, 9, 14, 17}:
            return attn_0_3_output == "1"
        elif position in {10, 20, 23, 28, 31}:
            return attn_0_3_output == "</s>"

    num_attn_1_4_pattern = select(attn_0_3_outputs, positions, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_5_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(mlp_0_3_output, token):
        if mlp_0_3_output in {
            0,
            1,
            4,
            5,
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
            20,
            21,
            22,
            23,
            24,
            26,
            27,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            39,
        }:
            return token == ""
        elif mlp_0_3_output in {2, 19}:
            return token == "<s>"
        elif mlp_0_3_output in {3, 7}:
            return token == "</s>"
        elif mlp_0_3_output in {6}:
            return token == "2"
        elif mlp_0_3_output in {25, 28}:
            return token == "<pad>"
        elif mlp_0_3_output in {38}:
            return token == "1"

    num_attn_1_5_pattern = select(tokens, mlp_0_3_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_6_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(position, attn_0_3_output):
        if position in {
            0,
            11,
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
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            39,
        }:
            return attn_0_3_output == ""
        elif position in {1, 2, 3, 4}:
            return attn_0_3_output == "<s>"
        elif position in {5, 6, 7, 8, 9, 10, 38}:
            return attn_0_3_output == "2"
        elif position in {16, 12, 14, 15}:
            return attn_0_3_output == "</s>"

    num_attn_1_6_pattern = select(attn_0_3_outputs, positions, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_5_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(num_mlp_0_0_output, num_mlp_0_3_output):
        if num_mlp_0_0_output in {0, 33, 20, 23}:
            return num_mlp_0_3_output == 36
        elif num_mlp_0_0_output in {1, 31}:
            return num_mlp_0_3_output == 33
        elif num_mlp_0_0_output in {2, 30}:
            return num_mlp_0_3_output == 32
        elif num_mlp_0_0_output in {3, 14, 25, 27, 28}:
            return num_mlp_0_3_output == 0
        elif num_mlp_0_0_output in {4}:
            return num_mlp_0_3_output == 16
        elif num_mlp_0_0_output in {5, 6}:
            return num_mlp_0_3_output == 14
        elif num_mlp_0_0_output in {7}:
            return num_mlp_0_3_output == 30
        elif num_mlp_0_0_output in {8, 38}:
            return num_mlp_0_3_output == 3
        elif num_mlp_0_0_output in {16, 9, 19, 17}:
            return num_mlp_0_3_output == 11
        elif num_mlp_0_0_output in {10, 26, 12, 29}:
            return num_mlp_0_3_output == 22
        elif num_mlp_0_0_output in {11, 22}:
            return num_mlp_0_3_output == 15
        elif num_mlp_0_0_output in {24, 21, 13}:
            return num_mlp_0_3_output == 18
        elif num_mlp_0_0_output in {15}:
            return num_mlp_0_3_output == 6
        elif num_mlp_0_0_output in {32, 18, 34}:
            return num_mlp_0_3_output == 13
        elif num_mlp_0_0_output in {35}:
            return num_mlp_0_3_output == 23
        elif num_mlp_0_0_output in {36}:
            return num_mlp_0_3_output == 25
        elif num_mlp_0_0_output in {37}:
            return num_mlp_0_3_output == 17
        elif num_mlp_0_0_output in {39}:
            return num_mlp_0_3_output == 21

    num_attn_1_7_pattern = select(
        num_mlp_0_3_outputs, num_mlp_0_0_outputs, num_predicate_1_7
    )
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_5_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(mlp_0_0_output, attn_1_6_output):
        key = (mlp_0_0_output, attn_1_6_output)
        return 16

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(mlp_0_0_outputs, attn_1_6_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_3_output, attn_0_5_output):
        key = (attn_1_3_output, attn_0_5_output)
        return 8

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_3_outputs, attn_0_5_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_1_2_output, attn_0_7_output):
        key = (attn_1_2_output, attn_0_7_output)
        return 37

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_0_7_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(num_mlp_0_0_output, position):
        key = (num_mlp_0_0_output, position)
        if key in {
            (0, 7),
            (0, 13),
            (0, 28),
            (1, 7),
            (1, 13),
            (1, 28),
            (2, 7),
            (2, 13),
            (2, 28),
            (3, 7),
            (3, 13),
            (3, 28),
            (4, 7),
            (4, 13),
            (4, 28),
            (5, 7),
            (5, 13),
            (5, 28),
            (6, 7),
            (6, 13),
            (6, 28),
            (7, 7),
            (7, 13),
            (7, 28),
            (8, 7),
            (8, 13),
            (8, 28),
            (9, 7),
            (9, 13),
            (10, 7),
            (10, 13),
            (10, 28),
            (11, 7),
            (11, 13),
            (11, 28),
            (11, 37),
            (12, 7),
            (12, 13),
            (12, 28),
            (13, 13),
            (14, 7),
            (14, 13),
            (14, 28),
            (15, 7),
            (15, 13),
            (15, 28),
            (15, 37),
            (16, 7),
            (16, 13),
            (17, 7),
            (17, 13),
            (17, 28),
            (18, 7),
            (18, 13),
            (18, 28),
            (19, 7),
            (19, 13),
            (19, 28),
            (20, 13),
            (21, 2),
            (21, 3),
            (21, 7),
            (21, 13),
            (21, 20),
            (21, 24),
            (21, 26),
            (21, 27),
            (21, 28),
            (21, 30),
            (21, 31),
            (21, 32),
            (21, 33),
            (21, 34),
            (21, 35),
            (21, 36),
            (21, 37),
            (21, 38),
            (21, 39),
            (22, 7),
            (22, 13),
            (22, 28),
            (23, 7),
            (23, 13),
            (23, 28),
            (24, 7),
            (24, 13),
            (24, 28),
            (24, 37),
            (25, 7),
            (25, 13),
            (25, 28),
            (25, 37),
            (26, 7),
            (26, 13),
            (26, 28),
            (27, 7),
            (27, 13),
            (28, 7),
            (28, 13),
            (28, 28),
            (29, 7),
            (29, 13),
            (30, 7),
            (30, 13),
            (30, 28),
            (31, 7),
            (31, 13),
            (32, 7),
            (32, 13),
            (32, 28),
            (33, 7),
            (33, 13),
            (33, 28),
            (34, 7),
            (34, 13),
            (34, 28),
            (35, 7),
            (35, 13),
            (35, 28),
            (36, 7),
            (36, 13),
            (36, 28),
            (37, 7),
            (37, 13),
            (38, 7),
            (38, 13),
            (38, 28),
            (39, 7),
            (39, 13),
            (39, 28),
            (39, 30),
            (39, 37),
            (39, 39),
        }:
            return 12
        elif key in {
            (0, 8),
            (0, 15),
            (0, 27),
            (1, 8),
            (1, 15),
            (1, 27),
            (2, 8),
            (2, 15),
            (2, 27),
            (3, 8),
            (3, 15),
            (3, 27),
            (4, 8),
            (4, 15),
            (5, 8),
            (5, 15),
            (5, 27),
            (6, 8),
            (6, 15),
            (6, 27),
            (7, 8),
            (7, 15),
            (7, 27),
            (8, 8),
            (8, 15),
            (8, 27),
            (9, 8),
            (9, 15),
            (9, 27),
            (9, 30),
            (9, 38),
            (10, 8),
            (10, 15),
            (10, 27),
            (11, 8),
            (11, 15),
            (11, 27),
            (12, 8),
            (12, 15),
            (12, 27),
            (13, 8),
            (13, 15),
            (13, 27),
            (14, 8),
            (14, 15),
            (14, 27),
            (15, 8),
            (15, 15),
            (15, 27),
            (16, 8),
            (16, 15),
            (16, 27),
            (17, 8),
            (17, 15),
            (17, 27),
            (18, 8),
            (18, 15),
            (19, 8),
            (19, 15),
            (19, 27),
            (20, 8),
            (20, 15),
            (20, 27),
            (21, 8),
            (21, 15),
            (22, 8),
            (22, 15),
            (22, 27),
            (23, 8),
            (23, 15),
            (23, 27),
            (24, 8),
            (24, 15),
            (24, 27),
            (25, 8),
            (25, 15),
            (25, 21),
            (25, 27),
            (25, 30),
            (25, 36),
            (25, 38),
            (26, 8),
            (26, 15),
            (26, 27),
            (27, 8),
            (27, 15),
            (27, 27),
            (28, 8),
            (28, 15),
            (28, 27),
            (29, 8),
            (29, 15),
            (29, 27),
            (30, 8),
            (30, 15),
            (30, 27),
            (31, 8),
            (31, 15),
            (31, 27),
            (32, 8),
            (32, 15),
            (32, 27),
            (33, 8),
            (33, 15),
            (33, 27),
            (34, 8),
            (34, 15),
            (34, 27),
            (35, 8),
            (35, 15),
            (35, 27),
            (36, 8),
            (36, 15),
            (36, 27),
            (37, 8),
            (37, 15),
            (37, 27),
            (38, 8),
            (38, 15),
            (39, 8),
            (39, 15),
            (39, 27),
        }:
            return 31
        elif key in {(7, 5), (26, 5), (26, 9), (26, 26), (33, 5)}:
            return 1
        return 37

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(num_mlp_0_0_outputs, positions)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_7_output, num_attn_1_2_output):
        key = (num_attn_0_7_output, num_attn_1_2_output)
        if key in {(0, 1), (1, 0)}:
            return 30
        elif key in {(0, 0)}:
            return 38
        return 14

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_7_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_4_output, num_attn_1_3_output):
        key = (num_attn_1_4_output, num_attn_1_3_output)
        return 30

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_5_output):
        key = num_attn_1_5_output
        return 14

    num_mlp_1_2_outputs = [num_mlp_1_2(k0) for k0 in num_attn_1_5_outputs]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_6_output, num_attn_1_1_output):
        key = (num_attn_1_6_output, num_attn_1_1_output)
        return 4

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_position, k_position):
        if q_position in {0, 34}:
            return k_position == 26
        elif q_position in {32, 1, 20, 17}:
            return k_position == 12
        elif q_position in {2, 27, 4}:
            return k_position == 3
        elif q_position in {3, 30}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {9, 6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 5
        elif q_position in {8, 16, 11, 19}:
            return k_position == 6
        elif q_position in {10, 38}:
            return k_position == 13
        elif q_position in {12, 14}:
            return k_position == 10
        elif q_position in {13, 15}:
            return k_position == 17
        elif q_position in {18, 36}:
            return k_position == 15
        elif q_position in {21}:
            return k_position == 22
        elif q_position in {24, 26, 22, 23}:
            return k_position == 28
        elif q_position in {25}:
            return k_position == 23
        elif q_position in {28, 29}:
            return k_position == 0
        elif q_position in {31}:
            return k_position == 35
        elif q_position in {33}:
            return k_position == 16
        elif q_position in {35}:
            return k_position == 39
        elif q_position in {37}:
            return k_position == 18
        elif q_position in {39}:
            return k_position == 34

    attn_2_0_pattern = select_closest(positions, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(position, mlp_0_2_output):
        if position in {0}:
            return mlp_0_2_output == 31
        elif position in {1}:
            return mlp_0_2_output == 37
        elif position in {2, 38, 10, 12, 14, 22, 25, 26, 27}:
            return mlp_0_2_output == 26
        elif position in {33, 3, 7, 23, 30}:
            return mlp_0_2_output == 25
        elif position in {4, 36, 6, 37, 11, 13, 24}:
            return mlp_0_2_output == 6
        elif position in {18, 5}:
            return mlp_0_2_output == 9
        elif position in {8, 16}:
            return mlp_0_2_output == 10
        elif position in {9, 28}:
            return mlp_0_2_output == 7
        elif position in {34, 15}:
            return mlp_0_2_output == 29
        elif position in {17}:
            return mlp_0_2_output == 14
        elif position in {19, 20}:
            return mlp_0_2_output == 4
        elif position in {21}:
            return mlp_0_2_output == 5
        elif position in {29}:
            return mlp_0_2_output == 0
        elif position in {31}:
            return mlp_0_2_output == 15
        elif position in {32}:
            return mlp_0_2_output == 1
        elif position in {35}:
            return mlp_0_2_output == 17
        elif position in {39}:
            return mlp_0_2_output == 3

    attn_2_1_pattern = select_closest(mlp_0_2_outputs, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_position, k_position):
        if q_position in {0, 37, 27, 28, 29}:
            return k_position == 0
        elif q_position in {8, 1}:
            return k_position == 12
        elif q_position in {2}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4, 20, 21, 22, 25, 31}:
            return k_position == 28
        elif q_position in {36, 5, 30}:
            return k_position == 13
        elif q_position in {11, 13, 6}:
            return k_position == 10
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {32, 9}:
            return k_position == 6
        elif q_position in {10}:
            return k_position == 5
        elif q_position in {12}:
            return k_position == 14
        elif q_position in {18, 14}:
            return k_position == 7
        elif q_position in {15}:
            return k_position == 19
        elif q_position in {16}:
            return k_position == 18
        elif q_position in {17}:
            return k_position == 15
        elif q_position in {19}:
            return k_position == 27
        elif q_position in {24, 26, 23}:
            return k_position == 29
        elif q_position in {33}:
            return k_position == 21
        elif q_position in {34}:
            return k_position == 24
        elif q_position in {35}:
            return k_position == 30
        elif q_position in {38}:
            return k_position == 2
        elif q_position in {39}:
            return k_position == 25

    attn_2_2_pattern = select_closest(positions, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_0_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_token, k_token):
        if q_token in {"1", "4", "0"}:
            return k_token == ""
        elif q_token in {"2"}:
            return k_token == "0"
        elif q_token in {"3"}:
            return k_token == "1"
        elif q_token in {"</s>", "<s>"}:
            return k_token == "<s>"

    attn_2_3_pattern = select_closest(tokens, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_6_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_1_3_output, mlp_0_1_output):
        if attn_1_3_output in {"0"}:
            return mlp_0_1_output == 29
        elif attn_1_3_output in {"1"}:
            return mlp_0_1_output == 36
        elif attn_1_3_output in {"2"}:
            return mlp_0_1_output == 2
        elif attn_1_3_output in {"3"}:
            return mlp_0_1_output == 26
        elif attn_1_3_output in {"4"}:
            return mlp_0_1_output == 11
        elif attn_1_3_output in {"</s>", "<s>"}:
            return mlp_0_1_output == 38

    attn_2_4_pattern = select_closest(mlp_0_1_outputs, attn_1_3_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_0_7_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_0_output, num_mlp_0_0_output):
        if attn_0_0_output in {"0"}:
            return num_mlp_0_0_output == 30
        elif attn_0_0_output in {"1"}:
            return num_mlp_0_0_output == 27
        elif attn_0_0_output in {"2"}:
            return num_mlp_0_0_output == 36
        elif attn_0_0_output in {"<s>", "3"}:
            return num_mlp_0_0_output == 6
        elif attn_0_0_output in {"4"}:
            return num_mlp_0_0_output == 39
        elif attn_0_0_output in {"</s>"}:
            return num_mlp_0_0_output == 7

    attn_2_5_pattern = select_closest(
        num_mlp_0_0_outputs, attn_0_0_outputs, predicate_2_5
    )
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_2_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(q_token, k_token):
        if q_token in {"1", "0"}:
            return k_token == "3"
        elif q_token in {"2"}:
            return k_token == "</s>"
        elif q_token in {"</s>", "3"}:
            return k_token == ""
        elif q_token in {"4"}:
            return k_token == "1"
        elif q_token in {"<s>"}:
            return k_token == "<s>"

    attn_2_6_pattern = select_closest(tokens, tokens, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_5_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(q_mlp_0_3_output, k_mlp_0_3_output):
        if q_mlp_0_3_output in {0}:
            return k_mlp_0_3_output == 29
        elif q_mlp_0_3_output in {1, 34}:
            return k_mlp_0_3_output == 27
        elif q_mlp_0_3_output in {2, 19, 21, 14}:
            return k_mlp_0_3_output == 15
        elif q_mlp_0_3_output in {3}:
            return k_mlp_0_3_output == 9
        elif q_mlp_0_3_output in {4}:
            return k_mlp_0_3_output == 36
        elif q_mlp_0_3_output in {5}:
            return k_mlp_0_3_output == 1
        elif q_mlp_0_3_output in {8, 37, 6}:
            return k_mlp_0_3_output == 16
        elif q_mlp_0_3_output in {32, 38, 7, 11, 12, 30}:
            return k_mlp_0_3_output == 6
        elif q_mlp_0_3_output in {9}:
            return k_mlp_0_3_output == 31
        elif q_mlp_0_3_output in {17, 10, 20}:
            return k_mlp_0_3_output == 19
        elif q_mlp_0_3_output in {33, 29, 13, 15}:
            return k_mlp_0_3_output == 38
        elif q_mlp_0_3_output in {16}:
            return k_mlp_0_3_output == 14
        elif q_mlp_0_3_output in {18}:
            return k_mlp_0_3_output == 0
        elif q_mlp_0_3_output in {22}:
            return k_mlp_0_3_output == 37
        elif q_mlp_0_3_output in {36, 23}:
            return k_mlp_0_3_output == 8
        elif q_mlp_0_3_output in {24}:
            return k_mlp_0_3_output == 34
        elif q_mlp_0_3_output in {25, 31}:
            return k_mlp_0_3_output == 11
        elif q_mlp_0_3_output in {26}:
            return k_mlp_0_3_output == 24
        elif q_mlp_0_3_output in {27}:
            return k_mlp_0_3_output == 18
        elif q_mlp_0_3_output in {35, 28}:
            return k_mlp_0_3_output == 2
        elif q_mlp_0_3_output in {39}:
            return k_mlp_0_3_output == 35

    attn_2_7_pattern = select_closest(mlp_0_3_outputs, mlp_0_3_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_0_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_0_1_output, token):
        if mlp_0_1_output in {0, 32, 3, 4, 35, 36, 37, 38, 31}:
            return token == "0"
        elif mlp_0_1_output in {1}:
            return token == "</s>"
        elif mlp_0_1_output in {2}:
            return token == "2"
        elif mlp_0_1_output in {5}:
            return token == "<s>"
        elif mlp_0_1_output in {
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
            39,
        }:
            return token == ""
        elif mlp_0_1_output in {12}:
            return token == "<pad>"

    num_attn_2_0_pattern = select(tokens, mlp_0_1_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_6_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, attn_1_4_output):
        if position in {0, 32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 34, 36, 37, 38, 29, 30, 31}:
            return attn_1_4_output == "0"
        elif position in {1}:
            return attn_1_4_output == "1"
        elif position in {
            33,
            35,
            39,
            11,
            15,
            16,
            17,
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
            return attn_1_4_output == ""
        elif position in {12, 14}:
            return attn_1_4_output == "</s>"
        elif position in {13}:
            return attn_1_4_output == "<s>"
        elif position in {18}:
            return attn_1_4_output == "<pad>"

    num_attn_2_1_pattern = select(attn_1_4_outputs, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_4_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_5_output, attn_1_7_output):
        if attn_1_5_output in {"2", "</s>", "<s>", "0"}:
            return attn_1_7_output == "1"
        elif attn_1_5_output in {"1"}:
            return attn_1_7_output == "2"
        elif attn_1_5_output in {"4", "3"}:
            return attn_1_7_output == "</s>"

    num_attn_2_2_pattern = select(attn_1_7_outputs, attn_1_5_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_5_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(position, attn_1_4_output):
        if position in {0, 3}:
            return attn_1_4_output == "2"
        elif position in {1, 29}:
            return attn_1_4_output == "<s>"
        elif position in {2, 14, 15}:
            return attn_1_4_output == "</s>"
        elif position in {32, 33, 34, 4, 5, 36, 7, 8, 9, 10, 11, 12, 38, 39, 31}:
            return attn_1_4_output == "1"
        elif position in {13, 6}:
            return attn_1_4_output == "0"
        elif position in {
            35,
            37,
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
        }:
            return attn_1_4_output == ""

    num_attn_2_3_pattern = select(attn_1_4_outputs, positions, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_4_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_1_5_output, position):
        if attn_1_5_output in {"0"}:
            return position == 0
        elif attn_1_5_output in {"1", "<s>", "3"}:
            return position == 26
        elif attn_1_5_output in {"2"}:
            return position == 18
        elif attn_1_5_output in {"4"}:
            return position == 17
        elif attn_1_5_output in {"</s>"}:
            return position == 13

    num_attn_2_4_pattern = select(positions, attn_1_5_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_7_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_1_6_output, attn_0_5_output):
        if attn_1_6_output in {"</s>", "0", "<s>", "3", "1"}:
            return attn_0_5_output == "1"
        elif attn_1_6_output in {"2"}:
            return attn_0_5_output == ""
        elif attn_1_6_output in {"4"}:
            return attn_0_5_output == "2"

    num_attn_2_5_pattern = select(attn_0_5_outputs, attn_1_6_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_7_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_5_output, num_mlp_0_3_output):
        if attn_1_5_output in {"0"}:
            return num_mlp_0_3_output == 16
        elif attn_1_5_output in {"1"}:
            return num_mlp_0_3_output == 7
        elif attn_1_5_output in {"2"}:
            return num_mlp_0_3_output == 8
        elif attn_1_5_output in {"3"}:
            return num_mlp_0_3_output == 28
        elif attn_1_5_output in {"4"}:
            return num_mlp_0_3_output == 14
        elif attn_1_5_output in {"</s>"}:
            return num_mlp_0_3_output == 32
        elif attn_1_5_output in {"<s>"}:
            return num_mlp_0_3_output == 20

    num_attn_2_6_pattern = select(
        num_mlp_0_3_outputs, attn_1_5_outputs, num_predicate_2_6
    )
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_7_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(num_mlp_0_0_output, position):
        if num_mlp_0_0_output in {0, 32, 34, 5, 13, 14, 15, 20, 23, 27}:
            return position == 22
        elif num_mlp_0_0_output in {1, 18, 3}:
            return position == 25
        elif num_mlp_0_0_output in {2, 38}:
            return position == 7
        elif num_mlp_0_0_output in {
            4,
            6,
            7,
            8,
            10,
            11,
            16,
            17,
            19,
            22,
            24,
            25,
            26,
            28,
            30,
            31,
            33,
            37,
            39,
        }:
            return position == 0
        elif num_mlp_0_0_output in {9, 12, 29, 36}:
            return position == 23
        elif num_mlp_0_0_output in {35, 21}:
            return position == 26

    num_attn_2_7_pattern = select(positions, num_mlp_0_0_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_5_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(mlp_0_2_output, num_mlp_1_1_output):
        key = (mlp_0_2_output, num_mlp_1_1_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 15),
            (0, 16),
            (0, 17),
            (0, 18),
            (0, 19),
            (0, 20),
            (0, 21),
            (0, 22),
            (0, 23),
            (0, 24),
            (0, 25),
            (0, 26),
            (0, 27),
            (0, 29),
            (0, 30),
            (0, 31),
            (0, 32),
            (0, 33),
            (0, 34),
            (0, 35),
            (0, 36),
            (0, 37),
            (0, 38),
            (0, 39),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
            (6, 4),
            (6, 5),
            (6, 6),
            (6, 7),
            (6, 8),
            (6, 9),
            (6, 10),
            (6, 11),
            (6, 12),
            (6, 13),
            (6, 14),
            (6, 15),
            (6, 16),
            (6, 17),
            (6, 18),
            (6, 19),
            (6, 20),
            (6, 21),
            (6, 22),
            (6, 23),
            (6, 24),
            (6, 25),
            (6, 26),
            (6, 27),
            (6, 28),
            (6, 29),
            (6, 30),
            (6, 31),
            (6, 32),
            (6, 33),
            (6, 34),
            (6, 35),
            (6, 36),
            (6, 37),
            (6, 38),
            (6, 39),
            (7, 3),
            (7, 23),
            (7, 35),
            (12, 3),
            (12, 23),
            (12, 35),
            (22, 3),
            (22, 23),
            (22, 35),
            (23, 23),
            (27, 1),
            (27, 3),
            (27, 4),
            (27, 5),
            (27, 7),
            (27, 9),
            (27, 10),
            (27, 11),
            (27, 12),
            (27, 15),
            (27, 16),
            (27, 17),
            (27, 18),
            (27, 20),
            (27, 23),
            (27, 29),
            (27, 30),
            (27, 34),
            (27, 35),
            (27, 37),
            (27, 38),
            (27, 39),
            (31, 3),
            (31, 23),
            (31, 35),
            (33, 23),
            (34, 23),
        }:
            return 32
        elif key in {(27, 2), (27, 6), (27, 14), (27, 25), (27, 36), (33, 3), (33, 35)}:
            return 26
        return 11

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(mlp_0_2_outputs, num_mlp_1_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(num_mlp_1_1_output, attn_0_7_output):
        key = (num_mlp_1_1_output, attn_0_7_output)
        return 25

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(num_mlp_1_1_outputs, attn_0_7_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(mlp_0_2_output, num_mlp_1_1_output):
        key = (mlp_0_2_output, num_mlp_1_1_output)
        if key in {(14, 24), (14, 38)}:
            return 24
        return 35

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(mlp_0_2_outputs, num_mlp_1_1_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(mlp_1_0_output, num_mlp_0_2_output):
        key = (mlp_1_0_output, num_mlp_0_2_output)
        return 17

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(mlp_1_0_outputs, num_mlp_0_2_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_4_output, num_attn_1_2_output):
        key = (num_attn_2_4_output, num_attn_1_2_output)
        return 31

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_4_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_2_output, num_attn_1_5_output):
        key = (num_attn_1_2_output, num_attn_1_5_output)
        return 15

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_2_7_output, num_attn_1_7_output):
        key = (num_attn_2_7_output, num_attn_1_7_output)
        return 16

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_2_7_outputs, num_attn_1_7_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_2_1_output, num_attn_1_0_output):
        key = (num_attn_2_1_output, num_attn_1_0_output)
        return 23

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_1_0_outputs)
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
        ["<s>", "0", "3", "3", "3", "1", "3", "2", "4", "0", "0", "4", "2", "1", "</s>"]
    )
)