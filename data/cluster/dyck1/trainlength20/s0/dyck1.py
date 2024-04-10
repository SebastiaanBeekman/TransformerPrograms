import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys[: i + 1]) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys[: i + 1]] for i, q in enumerate(queries)]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/length/rasp/dyck1/trainlength20/s0/dyck1_weights.csv",
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
        if token in {"("}:
            return position == 7
        elif token in {")"}:
            return position == 11
        elif token in {"<s>"}:
            return position == 13

    attn_0_0_pattern = select_closest(positions, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {
            0,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            17,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
        }:
            return k_position == 13
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 19}:
            return k_position == 10
        elif q_position in {3}:
            return k_position == 19
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 35
        elif q_position in {6}:
            return k_position == 2
        elif q_position in {20, 14, 7}:
            return k_position == 11
        elif q_position in {8, 10}:
            return k_position == 6
        elif q_position in {16, 9, 39, 31}:
            return k_position == 15
        elif q_position in {11}:
            return k_position == 30
        elif q_position in {12, 13}:
            return k_position == 8
        elif q_position in {15}:
            return k_position == 34
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {21}:
            return k_position == 27

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 26
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 12, 13}:
            return k_position == 10
        elif q_position in {3, 4}:
            return k_position == 3
        elif q_position in {32, 26, 5}:
            return k_position == 39
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 11
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 18
        elif q_position in {10, 11}:
            return k_position == 9
        elif q_position in {38, 14, 17, 20, 28, 29}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 12
        elif q_position in {16}:
            return k_position == 14
        elif q_position in {18, 19}:
            return k_position == 16
        elif q_position in {35, 21, 39}:
            return k_position == 37
        elif q_position in {27, 22}:
            return k_position == 15
        elif q_position in {23}:
            return k_position == 33
        elif q_position in {24}:
            return k_position == 34
        elif q_position in {25}:
            return k_position == 32
        elif q_position in {30, 31}:
            return k_position == 30
        elif q_position in {33}:
            return k_position == 27
        elif q_position in {34}:
            return k_position == 25
        elif q_position in {36}:
            return k_position == 21
        elif q_position in {37}:
            return k_position == 35

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 37, 15, 21, 22, 24, 26, 27, 28}:
            return k_position == 13
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {11, 2, 10}:
            return k_position == 8
        elif q_position in {3}:
            return k_position == 34
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {16, 18, 5}:
            return k_position == 15
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 33
        elif q_position in {8}:
            return k_position == 6
        elif q_position in {9}:
            return k_position == 28
        elif q_position in {32, 38, 12, 13, 20, 29}:
            return k_position == 11
        elif q_position in {14}:
            return k_position == 12
        elif q_position in {17, 19}:
            return k_position == 14
        elif q_position in {39, 23}:
            return k_position == 22
        elif q_position in {25, 34}:
            return k_position == 26
        elif q_position in {30}:
            return k_position == 39
        elif q_position in {33, 31}:
            return k_position == 23
        elif q_position in {35}:
            return k_position == 32
        elif q_position in {36}:
            return k_position == 37

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 11

    attn_0_4_pattern = select_closest(positions, tokens, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(token, position):
        if token in {"("}:
            return position == 9
        elif token in {")"}:
            return position == 12
        elif token in {"<s>"}:
            return position == 33

    attn_0_5_pattern = select_closest(positions, tokens, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 2
        elif token in {"<s>"}:
            return position == 32

    attn_0_6_pattern = select_closest(positions, tokens, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 10
        elif token in {"<s>"}:
            return position == 9

    attn_0_7_pattern = select_closest(positions, tokens, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0, 16}:
            return k_position == 22
        elif q_position in {1}:
            return k_position == 19
        elif q_position in {2, 12, 14, 19, 20, 28}:
            return k_position == 30
        elif q_position in {9, 3}:
            return k_position == 15
        elif q_position in {26, 4, 29, 36}:
            return k_position == 13
        elif q_position in {34, 5, 22}:
            return k_position == 26
        elif q_position in {6}:
            return k_position == 17
        elif q_position in {7}:
            return k_position == 4
        elif q_position in {8}:
            return k_position == 24
        elif q_position in {10, 18}:
            return k_position == 29
        elif q_position in {11}:
            return k_position == 18
        elif q_position in {13}:
            return k_position == 28
        elif q_position in {15}:
            return k_position == 34
        elif q_position in {17}:
            return k_position == 39
        elif q_position in {21, 31}:
            return k_position == 16
        elif q_position in {35, 39, 23, 25, 27}:
            return k_position == 1
        elif q_position in {24, 32}:
            return k_position == 21
        elif q_position in {30}:
            return k_position == 33
        elif q_position in {33}:
            return k_position == 25
        elif q_position in {37, 38}:
            return k_position == 23

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(token, position):
        if token in {")", "("}:
            return position == 22
        elif token in {"<s>"}:
            return position == 36

    num_attn_0_1_pattern = select(positions, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"("}:
            return position == 6
        elif token in {")"}:
            return position == 29
        elif token in {"<s>"}:
            return position == 2

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {
            0,
            2,
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
            19,
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
            34,
            35,
            37,
            39,
        }:
            return token == ""
        elif position in {1, 33, 3, 4, 36, 6, 38, 25, 28}:
            return token == ")"

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
            6,
            8,
            9,
            10,
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
        }:
            return token == ""
        elif position in {3, 4, 5, 7, 22}:
            return token == "("
        elif position in {13}:
            return token == "<pad>"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(token, position):
        if token in {"("}:
            return position == 20
        elif token in {")"}:
            return position == 22
        elif token in {"<s>"}:
            return position == 0

    num_attn_0_5_pattern = select(positions, tokens, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {
            0,
            2,
            4,
            6,
            8,
            10,
            16,
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
            return token == ""
        elif position in {1, 3, 5, 38, 7}:
            return token == ")"
        elif position in {9, 11, 12, 13, 14, 15, 17, 18}:
            return token == "<s>"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {2, 15}:
            return k_position == 26
        elif q_position in {32, 3, 36, 10, 22, 26, 28}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 35
        elif q_position in {5}:
            return k_position == 13
        elif q_position in {27, 6}:
            return k_position == 24
        elif q_position in {7}:
            return k_position == 12
        elif q_position in {8}:
            return k_position == 8
        elif q_position in {9}:
            return k_position == 0
        elif q_position in {11}:
            return k_position == 30
        elif q_position in {12}:
            return k_position == 32
        elif q_position in {13}:
            return k_position == 23
        elif q_position in {14}:
            return k_position == 28
        elif q_position in {16, 34}:
            return k_position == 33
        elif q_position in {17}:
            return k_position == 18
        elif q_position in {18}:
            return k_position == 36
        elif q_position in {19}:
            return k_position == 39
        elif q_position in {38, 20, 21, 25, 29, 30}:
            return k_position == 5
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24, 35}:
            return k_position == 38
        elif q_position in {31}:
            return k_position == 20
        elif q_position in {33}:
            return k_position == 7
        elif q_position in {37}:
            return k_position == 37
        elif q_position in {39}:
            return k_position == 21

    num_attn_0_7_pattern = select(positions, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_7_output, attn_0_3_output):
        key = (attn_0_7_output, attn_0_3_output)
        return 10

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_7_outputs, attn_0_3_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_3_output, attn_0_2_output):
        key = (attn_0_3_output, attn_0_2_output)
        return 22

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_2_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_5_output, num_attn_0_0_output):
        key = (num_attn_0_5_output, num_attn_0_0_output)
        return 10

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_5_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_0_output):
        key = num_attn_0_0_output
        if key in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}:
            return 14
        return 22

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(token, attn_0_1_output):
        if token in {"<s>", "("}:
            return attn_0_1_output == ""
        elif token in {")"}:
            return attn_0_1_output == ")"

    attn_1_0_pattern = select_closest(attn_0_1_outputs, tokens, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_1_output, position):
        if attn_0_1_output in {"<s>", "("}:
            return position == 1
        elif attn_0_1_output in {")"}:
            return position == 13

    attn_1_1_pattern = select_closest(positions, attn_0_1_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_6_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(token, attn_0_7_output):
        if token in {"("}:
            return attn_0_7_output == ""
        elif token in {"<s>", ")"}:
            return attn_0_7_output == ")"

    attn_1_2_pattern = select_closest(attn_0_7_outputs, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_2_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 7
        elif token in {"<s>"}:
            return position == 30

    attn_1_3_pattern = select_closest(positions, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(attn_0_7_output, position):
        if attn_0_7_output in {")", "("}:
            return position == 1
        elif attn_0_7_output in {"<s>"}:
            return position == 33

    attn_1_4_pattern = select_closest(positions, attn_0_7_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_7_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 9
        elif token in {"<s>"}:
            return position == 34

    attn_1_5_pattern = select_closest(positions, tokens, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, tokens)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(attn_0_1_output, position):
        if attn_0_1_output in {"("}:
            return position == 1
        elif attn_0_1_output in {")"}:
            return position == 11
        elif attn_0_1_output in {"<s>"}:
            return position == 3

    attn_1_6_pattern = select_closest(positions, attn_0_1_outputs, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_6_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(attn_0_3_output, position):
        if attn_0_3_output in {"<s>", "("}:
            return position == 3
        elif attn_0_3_output in {")"}:
            return position == 12

    attn_1_7_pattern = select_closest(positions, attn_0_3_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_6_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(mlp_0_1_output, position):
        if mlp_0_1_output in {0, 32, 12, 15, 21, 31}:
            return position == 11
        elif mlp_0_1_output in {1, 25}:
            return position == 27
        elif mlp_0_1_output in {2, 14}:
            return position == 13
        elif mlp_0_1_output in {3, 37, 18, 28, 29}:
            return position == 9
        elif mlp_0_1_output in {33, 4, 6, 39, 23, 27}:
            return position == 10
        elif mlp_0_1_output in {9, 19, 5, 35}:
            return position == 7
        elif mlp_0_1_output in {7}:
            return position == 20
        elif mlp_0_1_output in {8, 26}:
            return position == 1
        elif mlp_0_1_output in {10}:
            return position == 28
        elif mlp_0_1_output in {34, 11, 13, 16, 24, 30}:
            return position == 14
        elif mlp_0_1_output in {17}:
            return position == 34
        elif mlp_0_1_output in {20, 36}:
            return position == 12
        elif mlp_0_1_output in {38, 22}:
            return position == 8

    num_attn_1_0_pattern = select(positions, mlp_0_1_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_2_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_0_output, mlp_0_0_output):
        if attn_0_0_output in {"("}:
            return mlp_0_0_output == 33
        elif attn_0_0_output in {")"}:
            return mlp_0_0_output == 21
        elif attn_0_0_output in {"<s>"}:
            return mlp_0_0_output == 39

    num_attn_1_1_pattern = select(mlp_0_0_outputs, attn_0_0_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_2_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(mlp_0_0_output, attn_0_6_output):
        if mlp_0_0_output in {0, 34, 3, 7, 39, 11, 19, 20, 25, 26, 29, 30}:
            return attn_0_6_output == ")"
        elif mlp_0_0_output in {1, 5, 6, 8, 9}:
            return attn_0_6_output == "<s>"
        elif mlp_0_0_output in {
            2,
            4,
            10,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            21,
            22,
            23,
            24,
            27,
            28,
            31,
            32,
            33,
            35,
            36,
            37,
            38,
        }:
            return attn_0_6_output == ""

    num_attn_1_2_pattern = select(attn_0_6_outputs, mlp_0_0_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_3_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(mlp_0_1_output, mlp_0_0_output):
        if mlp_0_1_output in {0, 14}:
            return mlp_0_0_output == 20
        elif mlp_0_1_output in {1}:
            return mlp_0_0_output == 38
        elif mlp_0_1_output in {2}:
            return mlp_0_0_output == 18
        elif mlp_0_1_output in {3}:
            return mlp_0_0_output == 36
        elif mlp_0_1_output in {4}:
            return mlp_0_0_output == 10
        elif mlp_0_1_output in {35, 11, 5}:
            return mlp_0_0_output == 34
        elif mlp_0_1_output in {12, 13, 6}:
            return mlp_0_0_output == 5
        elif mlp_0_1_output in {28, 30, 7}:
            return mlp_0_0_output == 16
        elif mlp_0_1_output in {8, 32}:
            return mlp_0_0_output == 3
        elif mlp_0_1_output in {9}:
            return mlp_0_0_output == 6
        elif mlp_0_1_output in {33, 10, 23, 25, 29}:
            return mlp_0_0_output == 2
        elif mlp_0_1_output in {16, 36, 15}:
            return mlp_0_0_output == 21
        elif mlp_0_1_output in {26, 17, 18, 19}:
            return mlp_0_0_output == 37
        elif mlp_0_1_output in {24, 20}:
            return mlp_0_0_output == 12
        elif mlp_0_1_output in {34, 37, 21}:
            return mlp_0_0_output == 7
        elif mlp_0_1_output in {22}:
            return mlp_0_0_output == 19
        elif mlp_0_1_output in {27, 31}:
            return mlp_0_0_output == 25
        elif mlp_0_1_output in {38}:
            return mlp_0_0_output == 11
        elif mlp_0_1_output in {39}:
            return mlp_0_0_output == 30

    num_attn_1_3_pattern = select(mlp_0_0_outputs, mlp_0_1_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_3_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(attn_0_7_output, mlp_0_0_output):
        if attn_0_7_output in {"("}:
            return mlp_0_0_output == 23
        elif attn_0_7_output in {")"}:
            return mlp_0_0_output == 10
        elif attn_0_7_output in {"<s>"}:
            return mlp_0_0_output == 9

    num_attn_1_4_pattern = select(mlp_0_0_outputs, attn_0_7_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_2_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(num_mlp_0_1_output, attn_0_3_output):
        if num_mlp_0_1_output in {
            0,
            1,
            2,
            3,
            4,
            6,
            7,
            10,
            12,
            14,
            15,
            18,
            20,
            22,
            23,
            24,
            25,
            32,
            33,
            35,
            36,
            38,
            39,
        }:
            return attn_0_3_output == ")"
        elif num_mlp_0_1_output in {
            34,
            5,
            8,
            9,
            11,
            13,
            16,
            17,
            19,
            21,
            26,
            27,
            28,
            31,
        }:
            return attn_0_3_output == "<s>"
        elif num_mlp_0_1_output in {37, 29, 30}:
            return attn_0_3_output == ""

    num_attn_1_5_pattern = select(
        attn_0_3_outputs, num_mlp_0_1_outputs, num_predicate_1_5
    )
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_3_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(num_mlp_0_1_output, position):
        if num_mlp_0_1_output in {0, 33, 35, 39, 8, 13, 18, 20, 28, 30, 31}:
            return position == 4
        elif num_mlp_0_1_output in {
            1,
            3,
            4,
            5,
            6,
            7,
            9,
            10,
            11,
            14,
            15,
            16,
            17,
            19,
            21,
            22,
            23,
            24,
            25,
            27,
            29,
            32,
            37,
            38,
        }:
            return position == 3
        elif num_mlp_0_1_output in {2, 34, 36, 12, 26}:
            return position == 2

    num_attn_1_6_pattern = select(positions, num_mlp_0_1_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_3_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(position, attn_0_0_output):
        if position in {0, 1, 34, 6, 9, 11, 13, 14, 15, 16, 17, 18, 19, 25, 30}:
            return attn_0_0_output == ""
        elif position in {2, 5, 7, 12, 20, 21, 22, 24, 26}:
            return attn_0_0_output == "<s>"
        elif position in {32, 33, 3, 4, 35, 36, 37, 8, 38, 10, 39, 23, 27, 28, 29, 31}:
            return attn_0_0_output == ")"

    num_attn_1_7_pattern = select(attn_0_0_outputs, positions, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_3_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_3_output, attn_0_2_output):
        key = (attn_0_3_output, attn_0_2_output)
        if key in {
            ("(", "("),
            ("(", ")"),
            ("(", "<s>"),
            (")", "("),
            ("<s>", "("),
            ("<s>", "<s>"),
        }:
            return 1
        return 4

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_2_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_7_output, attn_0_4_output):
        key = (attn_0_7_output, attn_0_4_output)
        if key in {
            ("(", "("),
            ("(", ")"),
            ("(", "<s>"),
            (")", "("),
            ("<s>", "("),
            ("<s>", "<s>"),
        }:
            return 31
        return 32

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_7_outputs, attn_0_4_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_1_output, num_attn_1_4_output):
        key = (num_attn_1_1_output, num_attn_1_4_output)
        return 9

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_4_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_1_output, num_attn_1_2_output):
        key = (num_attn_1_1_output, num_attn_1_2_output)
        return 3

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_7_output, mlp_1_0_output):
        if attn_0_7_output in {"<s>", "("}:
            return mlp_1_0_output == 4
        elif attn_0_7_output in {")"}:
            return mlp_1_0_output == 5

    attn_2_0_pattern = select_closest(mlp_1_0_outputs, attn_0_7_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_0_1_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_1_output, position):
        if attn_0_1_output in {"("}:
            return position == 3
        elif attn_0_1_output in {"<s>", ")"}:
            return position == 5

    attn_2_1_pattern = select_closest(positions, attn_0_1_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_7_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_3_output, position):
        if attn_0_3_output in {"("}:
            return position == 4
        elif attn_0_3_output in {")"}:
            return position == 5
        elif attn_0_3_output in {"<s>"}:
            return position == 1

    attn_2_2_pattern = select_closest(positions, attn_0_3_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_4_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(token, attn_0_3_output):
        if token in {"<s>", "("}:
            return attn_0_3_output == ""
        elif token in {")"}:
            return attn_0_3_output == ")"

    attn_2_3_pattern = select_closest(attn_0_3_outputs, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_3_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(mlp_1_1_output, position):
        if mlp_1_1_output in {0}:
            return position == 28
        elif mlp_1_1_output in {1, 3, 5, 18, 22}:
            return position == 5
        elif mlp_1_1_output in {2}:
            return position == 26
        elif mlp_1_1_output in {34, 4, 37, 21, 24, 25, 30, 31}:
            return position == 7
        elif mlp_1_1_output in {8, 6}:
            return position == 21
        elif mlp_1_1_output in {12, 7}:
            return position == 39
        elif mlp_1_1_output in {9}:
            return position == 29
        elif mlp_1_1_output in {10, 19}:
            return position == 36
        elif mlp_1_1_output in {11}:
            return position == 25
        elif mlp_1_1_output in {13}:
            return position == 31
        elif mlp_1_1_output in {14}:
            return position == 15
        elif mlp_1_1_output in {38, 15}:
            return position == 24
        elif mlp_1_1_output in {16}:
            return position == 32
        elif mlp_1_1_output in {17}:
            return position == 23
        elif mlp_1_1_output in {32, 20}:
            return position == 8
        elif mlp_1_1_output in {33, 35, 23, 27, 29}:
            return position == 9
        elif mlp_1_1_output in {26}:
            return position == 11
        elif mlp_1_1_output in {28, 39}:
            return position == 10
        elif mlp_1_1_output in {36}:
            return position == 6

    attn_2_4_pattern = select_closest(positions, mlp_1_1_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_3_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_6_output, attn_0_7_output):
        if attn_0_6_output in {"<s>", ")", "("}:
            return attn_0_7_output == ")"

    attn_2_5_pattern = select_closest(attn_0_7_outputs, attn_0_6_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_5_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_0_3_output, position):
        if attn_0_3_output in {"("}:
            return position == 3
        elif attn_0_3_output in {")"}:
            return position == 11
        elif attn_0_3_output in {"<s>"}:
            return position == 4

    attn_2_6_pattern = select_closest(positions, attn_0_3_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_5_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_0_3_output, position):
        if attn_0_3_output in {"<s>", "("}:
            return position == 5
        elif attn_0_3_output in {")"}:
            return position == 10

    attn_2_7_pattern = select_closest(positions, attn_0_3_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_5_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_1_output, mlp_0_1_output):
        if attn_1_1_output in {"("}:
            return mlp_0_1_output == 25
        elif attn_1_1_output in {")"}:
            return mlp_0_1_output == 19
        elif attn_1_1_output in {"<s>"}:
            return mlp_0_1_output == 1

    num_attn_2_0_pattern = select(mlp_0_1_outputs, attn_1_1_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, mlp_1_1_output):
        if position in {0, 27}:
            return mlp_1_1_output == 20
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
            18,
            19,
            22,
            25,
            35,
            38,
        }:
            return mlp_1_1_output == 32
        elif position in {33, 37, 39, 20, 21, 23, 24, 28}:
            return mlp_1_1_output == 27
        elif position in {26, 30}:
            return mlp_1_1_output == 30
        elif position in {29}:
            return mlp_1_1_output == 29
        elif position in {31}:
            return mlp_1_1_output == 17
        elif position in {32}:
            return mlp_1_1_output == 22
        elif position in {34}:
            return mlp_1_1_output == 23
        elif position in {36}:
            return mlp_1_1_output == 10

    num_attn_2_1_pattern = select(mlp_1_1_outputs, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_6_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(num_mlp_1_1_output, attn_0_2_output):
        if num_mlp_1_1_output in {
            0,
            2,
            3,
            4,
            6,
            7,
            8,
            11,
            12,
            16,
            17,
            18,
            22,
            25,
            26,
            27,
            28,
            30,
            35,
            37,
        }:
            return attn_0_2_output == ")"
        elif num_mlp_1_1_output in {
            32,
            1,
            33,
            34,
            36,
            5,
            38,
            39,
            9,
            10,
            13,
            14,
            19,
            20,
            23,
            29,
            31,
        }:
            return attn_0_2_output == ""
        elif num_mlp_1_1_output in {21, 15}:
            return attn_0_2_output == "<s>"
        elif num_mlp_1_1_output in {24}:
            return attn_0_2_output == "<pad>"

    num_attn_2_2_pattern = select(
        attn_0_2_outputs, num_mlp_1_1_outputs, num_predicate_2_2
    )
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_3_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(q_mlp_1_1_output, k_mlp_1_1_output):
        if q_mlp_1_1_output in {0, 5, 6, 39, 12, 14, 23, 24, 28}:
            return k_mlp_1_1_output == 14
        elif q_mlp_1_1_output in {1, 2, 3, 36, 18, 22, 31}:
            return k_mlp_1_1_output == 32
        elif q_mlp_1_1_output in {4, 13}:
            return k_mlp_1_1_output == 10
        elif q_mlp_1_1_output in {8, 11, 7}:
            return k_mlp_1_1_output == 9
        elif q_mlp_1_1_output in {9}:
            return k_mlp_1_1_output == 6
        elif q_mlp_1_1_output in {10, 21}:
            return k_mlp_1_1_output == 20
        elif q_mlp_1_1_output in {15}:
            return k_mlp_1_1_output == 35
        elif q_mlp_1_1_output in {16, 35, 27}:
            return k_mlp_1_1_output == 23
        elif q_mlp_1_1_output in {17, 34}:
            return k_mlp_1_1_output == 7
        elif q_mlp_1_1_output in {25, 19}:
            return k_mlp_1_1_output == 4
        elif q_mlp_1_1_output in {20}:
            return k_mlp_1_1_output == 21
        elif q_mlp_1_1_output in {26}:
            return k_mlp_1_1_output == 13
        elif q_mlp_1_1_output in {29}:
            return k_mlp_1_1_output == 28
        elif q_mlp_1_1_output in {30}:
            return k_mlp_1_1_output == 30
        elif q_mlp_1_1_output in {32}:
            return k_mlp_1_1_output == 8
        elif q_mlp_1_1_output in {33, 37}:
            return k_mlp_1_1_output == 17
        elif q_mlp_1_1_output in {38}:
            return k_mlp_1_1_output == 12

    num_attn_2_3_pattern = select(mlp_1_1_outputs, mlp_1_1_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_7_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(mlp_1_1_output, attn_1_7_output):
        if mlp_1_1_output in {
            0,
            1,
            2,
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
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            24,
            25,
            26,
            27,
            28,
            30,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
        }:
            return attn_1_7_output == ""
        elif mlp_1_1_output in {31, 29, 6, 22}:
            return attn_1_7_output == ")"
        elif mlp_1_1_output in {14}:
            return attn_1_7_output == "("
        elif mlp_1_1_output in {23}:
            return attn_1_7_output == "<pad>"

    num_attn_2_4_pattern = select(attn_1_7_outputs, mlp_1_1_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_3_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(mlp_0_0_output, mlp_1_0_output):
        if mlp_0_0_output in {0, 13}:
            return mlp_1_0_output == 24
        elif mlp_0_0_output in {1}:
            return mlp_1_0_output == 38
        elif mlp_0_0_output in {2, 23}:
            return mlp_1_0_output == 25
        elif mlp_0_0_output in {11, 19, 3, 38}:
            return mlp_1_0_output == 26
        elif mlp_0_0_output in {
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            12,
            17,
            18,
            20,
            21,
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
            return mlp_1_0_output == 1
        elif mlp_0_0_output in {16, 14}:
            return mlp_1_0_output == 23
        elif mlp_0_0_output in {15}:
            return mlp_1_0_output == 39
        elif mlp_0_0_output in {25, 22}:
            return mlp_1_0_output == 30
        elif mlp_0_0_output in {24}:
            return mlp_1_0_output == 12

    num_attn_2_5_pattern = select(mlp_1_0_outputs, mlp_0_0_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_2_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_6_output, attn_0_5_output):
        if attn_1_6_output in {"<s>", ")", "("}:
            return attn_0_5_output == ""

    num_attn_2_6_pattern = select(attn_0_5_outputs, attn_1_6_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_1_0_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(num_mlp_0_1_output, position):
        if num_mlp_0_1_output in {0, 38, 6, 7, 27}:
            return position == 1
        elif num_mlp_0_1_output in {1, 9, 17}:
            return position == 25
        elif num_mlp_0_1_output in {2}:
            return position == 16
        elif num_mlp_0_1_output in {3}:
            return position == 24
        elif num_mlp_0_1_output in {4}:
            return position == 26
        elif num_mlp_0_1_output in {5, 23}:
            return position == 7
        elif num_mlp_0_1_output in {8, 24}:
            return position == 6
        elif num_mlp_0_1_output in {10, 36, 30}:
            return position == 22
        elif num_mlp_0_1_output in {11}:
            return position == 18
        elif num_mlp_0_1_output in {12}:
            return position == 11
        elif num_mlp_0_1_output in {16, 34, 13}:
            return position == 15
        elif num_mlp_0_1_output in {14, 31}:
            return position == 13
        elif num_mlp_0_1_output in {15}:
            return position == 30
        elif num_mlp_0_1_output in {18, 20, 21}:
            return position == 3
        elif num_mlp_0_1_output in {19}:
            return position == 34
        elif num_mlp_0_1_output in {22}:
            return position == 35
        elif num_mlp_0_1_output in {25}:
            return position == 8
        elif num_mlp_0_1_output in {26}:
            return position == 33
        elif num_mlp_0_1_output in {28}:
            return position == 37
        elif num_mlp_0_1_output in {29}:
            return position == 29
        elif num_mlp_0_1_output in {32}:
            return position == 10
        elif num_mlp_0_1_output in {33}:
            return position == 27
        elif num_mlp_0_1_output in {35}:
            return position == 9
        elif num_mlp_0_1_output in {37}:
            return position == 17
        elif num_mlp_0_1_output in {39}:
            return position == 5

    num_attn_2_7_pattern = select(positions, num_mlp_0_1_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_2_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_2_output, attn_2_1_output):
        key = (attn_2_2_output, attn_2_1_output)
        if key in {("<s>", "<s>")}:
            return 14
        elif key in {(")", "("), (")", ")"), (")", "<s>"), ("<s>", ")")}:
            return 33
        return 0

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_2_outputs, attn_2_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_2_output, attn_2_4_output):
        key = (attn_2_2_output, attn_2_4_output)
        if key in {("<s>", "("), ("<s>", "<s>")}:
            return 5
        elif key in {("(", "(")}:
            return 19
        return 12

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_2_outputs, attn_2_4_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_0_output, num_attn_2_1_output):
        key = (num_attn_2_0_output, num_attn_2_1_output)
        return 23

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_0_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_3_output, num_attn_2_4_output):
        key = (num_attn_2_3_output, num_attn_2_4_output)
        return 20

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_2_4_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
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
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
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
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
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
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
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
            ")",
            ")",
            "(",
            ")",
            ")",
            ")",
            ")",
            ")",
            ")",
            ")",
            "(",
            "(",
            ")",
            "(",
            "(",
            "(",
            "(",
            "(",
            ")",
        ]
    )
)
