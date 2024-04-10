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
        "output/length/rasp/dyck1/trainlength20/s3/dyck1_weights.csv",
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
            return position == 1
        elif token in {")"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 38

    attn_0_0_pattern = select_closest(positions, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 37, 39, 20, 22}:
            return k_position == 9
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 6
        elif q_position in {3}:
            return k_position == 22
        elif q_position in {10, 4}:
            return k_position == 2
        elif q_position in {25, 11, 5}:
            return k_position == 13
        elif q_position in {8, 6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 29
        elif q_position in {32, 33, 34, 38, 9, 14, 23, 27}:
            return k_position == 15
        elif q_position in {35, 12, 21, 24, 28, 29, 30}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 36
        elif q_position in {15}:
            return k_position == 39
        elif q_position in {16}:
            return k_position == 12
        elif q_position in {17}:
            return k_position == 10
        elif q_position in {18}:
            return k_position == 16
        elif q_position in {19, 31}:
            return k_position == 17
        elif q_position in {26, 36}:
            return k_position == 19

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 32, 33, 34, 10, 23, 25}:
            return k_position == 9
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 12, 13}:
            return k_position == 10
        elif q_position in {3, 4}:
            return k_position == 2
        elif q_position in {24, 5}:
            return k_position == 23
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 5
        elif q_position in {8, 9}:
            return k_position == 6
        elif q_position in {11}:
            return k_position == 26
        elif q_position in {14, 15}:
            return k_position == 13
        elif q_position in {16, 17}:
            return k_position == 14
        elif q_position in {18, 19}:
            return k_position == 16
        elif q_position in {20}:
            return k_position == 39
        elif q_position in {21, 39}:
            return k_position == 33
        elif q_position in {22}:
            return k_position == 30
        elif q_position in {26, 27}:
            return k_position == 32
        elif q_position in {28}:
            return k_position == 17
        elif q_position in {29}:
            return k_position == 29
        elif q_position in {30}:
            return k_position == 37
        elif q_position in {38, 31}:
            return k_position == 28
        elif q_position in {35}:
            return k_position == 25
        elif q_position in {36}:
            return k_position == 35
        elif q_position in {37}:
            return k_position == 24

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 10
        elif token in {"<s>"}:
            return position == 9

    attn_0_3_pattern = select_closest(positions, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 36, 31}:
            return k_position == 19
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 12, 13, 14, 15}:
            return k_position == 10
        elif q_position in {3}:
            return k_position == 21
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {5, 38}:
            return k_position == 20
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {24, 7}:
            return k_position == 11
        elif q_position in {8}:
            return k_position == 4
        elif q_position in {9, 21}:
            return k_position == 25
        elif q_position in {10}:
            return k_position == 7
        elif q_position in {11}:
            return k_position == 8
        elif q_position in {16, 34, 28}:
            return k_position == 13
        elif q_position in {17, 18}:
            return k_position == 14
        elif q_position in {32, 19}:
            return k_position == 17
        elif q_position in {20}:
            return k_position == 22
        elif q_position in {37, 22}:
            return k_position == 33
        elif q_position in {23}:
            return k_position == 23
        elif q_position in {25, 29}:
            return k_position == 28
        elif q_position in {33, 26}:
            return k_position == 31
        elif q_position in {27}:
            return k_position == 27
        elif q_position in {30}:
            return k_position == 38
        elif q_position in {35}:
            return k_position == 39
        elif q_position in {39}:
            return k_position == 30

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0, 27, 28}:
            return k_position == 23
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 38
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5, 6}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9, 10}:
            return k_position == 8
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13, 14}:
            return k_position == 12
        elif q_position in {19, 15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18, 26}:
            return k_position == 17
        elif q_position in {20, 29}:
            return k_position == 34
        elif q_position in {21}:
            return k_position == 27
        elif q_position in {22}:
            return k_position == 36
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 37
        elif q_position in {32, 25, 38}:
            return k_position == 19
        elif q_position in {30}:
            return k_position == 28
        elif q_position in {31}:
            return k_position == 18
        elif q_position in {33, 36}:
            return k_position == 39
        elif q_position in {34, 39}:
            return k_position == 32
        elif q_position in {35}:
            return k_position == 24
        elif q_position in {37}:
            return k_position == 31

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(token, position):
        if token in {"("}:
            return position == 19
        elif token in {")"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 14

    attn_0_6_pattern = select_closest(positions, tokens, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(token, position):
        if token in {"("}:
            return position == 27
        elif token in {")"}:
            return position == 2
        elif token in {"<s>"}:
            return position == 35

    attn_0_7_pattern = select_closest(positions, tokens, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"("}:
            return position == 32
        elif token in {")"}:
            return position == 35
        elif token in {"<s>"}:
            return position == 17

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 7
        elif q_position in {1, 22}:
            return k_position == 39
        elif q_position in {16, 2, 29}:
            return k_position == 24
        elif q_position in {9, 3}:
            return k_position == 3
        elif q_position in {4, 21, 30}:
            return k_position == 29
        elif q_position in {25, 5}:
            return k_position == 36
        elif q_position in {6, 7}:
            return k_position == 8
        elif q_position in {8, 12, 37}:
            return k_position == 21
        elif q_position in {10, 31}:
            return k_position == 33
        elif q_position in {34, 11}:
            return k_position == 25
        elif q_position in {32, 33, 13, 15}:
            return k_position == 17
        elif q_position in {14}:
            return k_position == 35
        elif q_position in {17}:
            return k_position == 23
        elif q_position in {18, 26}:
            return k_position == 31
        elif q_position in {19, 20}:
            return k_position == 30
        elif q_position in {35, 38, 23, 27, 28}:
            return k_position == 1
        elif q_position in {24}:
            return k_position == 38
        elif q_position in {36}:
            return k_position == 18
        elif q_position in {39}:
            return k_position == 13

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"("}:
            return position == 11
        elif token in {")"}:
            return position == 26
        elif token in {"<s>"}:
            return position == 29

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
            4,
            6,
            7,
            8,
            10,
            12,
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
        elif position in {1, 3, 5, 31}:
            return token == ")"
        elif position in {9, 13}:
            return token == "<s>"
        elif position in {11}:
            return token == "<pad>"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(q_position, k_position):
        if q_position in {0}:
            return k_position == 6
        elif q_position in {1, 14}:
            return k_position == 26
        elif q_position in {2, 11, 38}:
            return k_position == 34
        elif q_position in {10, 3}:
            return k_position == 11
        elif q_position in {4, 37}:
            return k_position == 20
        elif q_position in {20, 5}:
            return k_position == 33
        elif q_position in {35, 6}:
            return k_position == 39
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {8, 13, 22}:
            return k_position == 17
        elif q_position in {9, 12}:
            return k_position == 24
        elif q_position in {29, 15}:
            return k_position == 38
        elif q_position in {16}:
            return k_position == 22
        elif q_position in {17}:
            return k_position == 28
        elif q_position in {33, 18, 31}:
            return k_position == 31
        elif q_position in {19}:
            return k_position == 37
        elif q_position in {21}:
            return k_position == 27
        elif q_position in {23}:
            return k_position == 16
        elif q_position in {24, 32}:
            return k_position == 23
        elif q_position in {25}:
            return k_position == 1
        elif q_position in {26, 27}:
            return k_position == 29
        elif q_position in {28}:
            return k_position == 14
        elif q_position in {30, 39}:
            return k_position == 25
        elif q_position in {34}:
            return k_position == 21
        elif q_position in {36}:
            return k_position == 35

    num_attn_0_4_pattern = select(positions, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {
            0,
            2,
            4,
            6,
            8,
            10,
            14,
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
            39,
        }:
            return token == ""
        elif position in {1, 9, 11, 12, 13, 15}:
            return token == "<s>"
        elif position in {24, 3, 5, 7}:
            return token == ")"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(token, position):
        if token in {"("}:
            return position == 8
        elif token in {")"}:
            return position == 33
        elif token in {"<s>"}:
            return position == 28

    num_attn_0_6_pattern = select(positions, tokens, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {
            0,
            1,
            5,
            7,
            9,
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
        }:
            return token == ""
        elif position in {2, 3, 4, 6, 8, 23}:
            return token == ")"
        elif position in {10}:
            return token == "<s>"
        elif position in {18}:
            return token == "<pad>"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_3_output, attn_0_2_output):
        key = (attn_0_3_output, attn_0_2_output)
        if key in {(")", "("), (")", ")"), (")", "<s>"), ("<s>", ")"), ("<s>", "<s>")}:
            return 2
        elif key in {("<s>", "(")}:
            return 7
        return 12

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_2_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_0_output, attn_0_5_output):
        key = (attn_0_0_output, attn_0_5_output)
        if key in {("<s>", ")")}:
            return 7
        elif key in {(")", ")"), (")", "<s>")}:
            return 2
        return 18

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_5_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output, num_attn_0_0_output):
        key = (num_attn_0_1_output, num_attn_0_0_output)
        return 20

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_7_output, num_attn_0_6_output):
        key = (num_attn_0_7_output, num_attn_0_6_output)
        if key in {
            (25, 0),
            (26, 0),
            (27, 0),
            (28, 0),
            (29, 0),
            (30, 0),
            (31, 0),
            (31, 1),
            (32, 0),
            (32, 1),
            (33, 0),
            (33, 1),
            (34, 0),
            (34, 1),
            (35, 0),
            (35, 1),
            (36, 0),
            (36, 1),
            (37, 0),
            (37, 1),
            (37, 2),
            (38, 0),
            (38, 1),
            (38, 2),
            (39, 0),
            (39, 1),
            (39, 2),
        }:
            return 0
        return 38

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_7_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(mlp_0_0_output, attn_0_5_output):
        if mlp_0_0_output in {
            0,
            3,
            4,
            6,
            7,
            8,
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
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            36,
            37,
            38,
            39,
        }:
            return attn_0_5_output == ""
        elif mlp_0_0_output in {1}:
            return attn_0_5_output == "<s>"
        elif mlp_0_0_output in {2, 35, 5, 9, 12, 20, 26}:
            return attn_0_5_output == ")"

    attn_1_0_pattern = select_closest(attn_0_5_outputs, mlp_0_0_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_2_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(token, attn_0_4_output):
        if token in {")", "("}:
            return attn_0_4_output == ")"
        elif token in {"<s>"}:
            return attn_0_4_output == ""

    attn_1_1_pattern = select_closest(attn_0_4_outputs, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_3_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_token, k_token):
        if q_token in {"("}:
            return k_token == "("
        elif q_token in {")", "<s>"}:
            return k_token == ""

    attn_1_2_pattern = select_closest(tokens, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_5_output, position):
        if attn_0_5_output in {"(", "<s>"}:
            return position == 1
        elif attn_0_5_output in {")"}:
            return position == 11

    attn_1_3_pattern = select_closest(positions, attn_0_5_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(token, attn_0_2_output):
        if token in {")", "("}:
            return attn_0_2_output == ")"
        elif token in {"<s>"}:
            return attn_0_2_output == ""

    attn_1_4_pattern = select_closest(attn_0_2_outputs, tokens, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, tokens)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(attn_0_7_output, position):
        if attn_0_7_output in {"("}:
            return position == 17
        elif attn_0_7_output in {")"}:
            return position == 14
        elif attn_0_7_output in {"<s>"}:
            return position == 1

    attn_1_5_pattern = select_closest(positions, attn_0_7_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_1_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 3
        elif token in {"<s>"}:
            return position == 17

    attn_1_6_pattern = select_closest(positions, tokens, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, tokens)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(attn_0_5_output, position):
        if attn_0_5_output in {"("}:
            return position == 5
        elif attn_0_5_output in {")"}:
            return position == 15
        elif attn_0_5_output in {"<s>"}:
            return position == 7

    attn_1_7_pattern = select_closest(positions, attn_0_5_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, tokens)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(num_mlp_0_1_output, position):
        if num_mlp_0_1_output in {0, 27, 28, 7}:
            return position == 18
        elif num_mlp_0_1_output in {32, 1, 3, 35, 5, 37, 38, 13, 14, 16, 31}:
            return position == 9
        elif num_mlp_0_1_output in {2}:
            return position == 15
        elif num_mlp_0_1_output in {4}:
            return position == 4
        elif num_mlp_0_1_output in {29, 6}:
            return position == 13
        elif num_mlp_0_1_output in {8}:
            return position == 35
        elif num_mlp_0_1_output in {9, 36}:
            return position == 24
        elif num_mlp_0_1_output in {10}:
            return position == 1
        elif num_mlp_0_1_output in {11, 15, 20, 21, 22}:
            return position == 11
        elif num_mlp_0_1_output in {12, 19, 23, 24, 25}:
            return position == 14
        elif num_mlp_0_1_output in {17, 18, 26}:
            return position == 10
        elif num_mlp_0_1_output in {30}:
            return position == 28
        elif num_mlp_0_1_output in {33}:
            return position == 16
        elif num_mlp_0_1_output in {34}:
            return position == 8
        elif num_mlp_0_1_output in {39}:
            return position == 7

    num_attn_1_0_pattern = select(positions, num_mlp_0_1_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_6_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_2_output, mlp_0_0_output):
        if attn_0_2_output in {"("}:
            return mlp_0_0_output == 17
        elif attn_0_2_output in {")"}:
            return mlp_0_0_output == 9
        elif attn_0_2_output in {"<s>"}:
            return mlp_0_0_output == 20

    num_attn_1_1_pattern = select(mlp_0_0_outputs, attn_0_2_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_4_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_5_output, token):
        if attn_0_5_output in {"("}:
            return token == ""
        elif attn_0_5_output in {")", "<s>"}:
            return token == ")"

    num_attn_1_2_pattern = select(tokens, attn_0_5_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_3_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_3_output, mlp_0_0_output):
        if attn_0_3_output in {"(", "<s>"}:
            return mlp_0_0_output == 2
        elif attn_0_3_output in {")"}:
            return mlp_0_0_output == 6

    num_attn_1_3_pattern = select(mlp_0_0_outputs, attn_0_3_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_7_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(q_mlp_0_0_output, k_mlp_0_0_output):
        if q_mlp_0_0_output in {0}:
            return k_mlp_0_0_output == 10
        elif q_mlp_0_0_output in {32, 1, 4, 20}:
            return k_mlp_0_0_output == 7
        elif q_mlp_0_0_output in {2}:
            return k_mlp_0_0_output == 19
        elif q_mlp_0_0_output in {33, 3, 35, 6, 38, 39, 9, 12, 18, 21, 22, 23, 29}:
            return k_mlp_0_0_output == 2
        elif q_mlp_0_0_output in {36, 5, 11, 13, 14, 15, 16, 17, 19, 25}:
            return k_mlp_0_0_output == 5
        elif q_mlp_0_0_output in {28, 7}:
            return k_mlp_0_0_output == 25
        elif q_mlp_0_0_output in {8}:
            return k_mlp_0_0_output == 16
        elif q_mlp_0_0_output in {10}:
            return k_mlp_0_0_output == 11
        elif q_mlp_0_0_output in {24}:
            return k_mlp_0_0_output == 8
        elif q_mlp_0_0_output in {26, 37, 30}:
            return k_mlp_0_0_output == 9
        elif q_mlp_0_0_output in {27}:
            return k_mlp_0_0_output == 24
        elif q_mlp_0_0_output in {31}:
            return k_mlp_0_0_output == 30
        elif q_mlp_0_0_output in {34}:
            return k_mlp_0_0_output == 33

    num_attn_1_4_pattern = select(mlp_0_0_outputs, mlp_0_0_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_3_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(num_mlp_0_0_output, attn_0_5_output):
        if num_mlp_0_0_output in {
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            9,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            21,
            22,
            24,
            25,
            26,
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
        }:
            return attn_0_5_output == ""
        elif num_mlp_0_0_output in {8, 10, 11, 12, 23, 27}:
            return attn_0_5_output == "<s>"
        elif num_mlp_0_0_output in {33, 20, 39}:
            return attn_0_5_output == ")"

    num_attn_1_5_pattern = select(
        attn_0_5_outputs, num_mlp_0_0_outputs, num_predicate_1_5
    )
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_7_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(mlp_0_0_output, position):
        if mlp_0_0_output in {0}:
            return position == 4
        elif mlp_0_0_output in {32, 1, 5, 37, 10, 16, 17, 19}:
            return position == 22
        elif mlp_0_0_output in {2, 35, 7, 15, 22, 25, 31}:
            return position == 12
        elif mlp_0_0_output in {8, 27, 34, 3}:
            return position == 1
        elif mlp_0_0_output in {4}:
            return position == 8
        elif mlp_0_0_output in {38, 6}:
            return position == 33
        elif mlp_0_0_output in {9}:
            return position == 9
        elif mlp_0_0_output in {11, 18, 23, 28, 29}:
            return position == 18
        elif mlp_0_0_output in {12}:
            return position == 6
        elif mlp_0_0_output in {13}:
            return position == 25
        elif mlp_0_0_output in {14}:
            return position == 31
        elif mlp_0_0_output in {20, 36, 39}:
            return position == 26
        elif mlp_0_0_output in {21}:
            return position == 17
        elif mlp_0_0_output in {24}:
            return position == 16
        elif mlp_0_0_output in {26}:
            return position == 15
        elif mlp_0_0_output in {30}:
            return position == 13
        elif mlp_0_0_output in {33}:
            return position == 27

    num_attn_1_6_pattern = select(positions, mlp_0_0_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_2_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(num_mlp_0_1_output, attn_0_6_output):
        if num_mlp_0_1_output in {0, 9, 6}:
            return attn_0_6_output == "("
        elif num_mlp_0_1_output in {
            1,
            2,
            3,
            4,
            5,
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
            return attn_0_6_output == ""

    num_attn_1_7_pattern = select(
        attn_0_6_outputs, num_mlp_0_1_outputs, num_predicate_1_7
    )
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_2_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_5_output, attn_1_2_output):
        key = (attn_0_5_output, attn_1_2_output)
        if key in {(")", "("), ("<s>", "(")}:
            return 7
        elif key in {("(", "("), ("(", ")")}:
            return 36
        elif key in {("(", "<s>")}:
            return 27
        return 2

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_5_outputs, attn_1_2_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_5_output, attn_0_2_output):
        key = (attn_0_5_output, attn_0_2_output)
        if key in {("(", "(")}:
            return 14
        return 13

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_5_outputs, attn_0_2_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_7_output, num_attn_1_5_output):
        key = (num_attn_1_7_output, num_attn_1_5_output)
        if key in {
            (18, 0),
            (19, 0),
            (20, 0),
            (21, 0),
            (22, 0),
            (23, 0),
            (24, 0),
            (25, 0),
            (26, 0),
            (27, 0),
            (28, 0),
            (29, 0),
            (30, 0),
            (31, 0),
            (32, 0),
            (33, 0),
            (34, 0),
            (35, 0),
            (36, 0),
            (37, 0),
            (38, 0),
            (39, 0),
            (40, 0),
            (40, 1),
            (41, 0),
            (41, 1),
            (42, 0),
            (42, 1),
            (43, 0),
            (43, 1),
            (44, 0),
            (44, 1),
            (45, 0),
            (45, 1),
            (46, 0),
            (46, 1),
            (47, 0),
            (47, 1),
            (48, 0),
            (48, 1),
            (49, 0),
            (49, 1),
            (50, 0),
            (50, 1),
            (51, 0),
            (51, 1),
            (52, 0),
            (52, 1),
            (53, 0),
            (53, 1),
            (54, 0),
            (54, 1),
            (55, 0),
            (55, 1),
            (56, 0),
            (56, 1),
            (57, 0),
            (57, 1),
            (58, 0),
            (58, 1),
            (59, 0),
            (59, 1),
            (60, 0),
            (60, 1),
            (61, 0),
            (61, 1),
            (62, 0),
            (62, 1),
            (62, 2),
            (63, 0),
            (63, 1),
            (63, 2),
            (64, 0),
            (64, 1),
            (64, 2),
            (65, 0),
            (65, 1),
            (65, 2),
            (66, 0),
            (66, 1),
            (66, 2),
            (67, 0),
            (67, 1),
            (67, 2),
            (68, 0),
            (68, 1),
            (68, 2),
            (69, 0),
            (69, 1),
            (69, 2),
            (70, 0),
            (70, 1),
            (70, 2),
            (71, 0),
            (71, 1),
            (71, 2),
            (72, 0),
            (72, 1),
            (72, 2),
            (73, 0),
            (73, 1),
            (73, 2),
            (74, 0),
            (74, 1),
            (74, 2),
            (75, 0),
            (75, 1),
            (75, 2),
            (76, 0),
            (76, 1),
            (76, 2),
            (77, 0),
            (77, 1),
            (77, 2),
            (78, 0),
            (78, 1),
            (78, 2),
            (79, 0),
            (79, 1),
            (79, 2),
        }:
            return 12
        return 32

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_7_output, num_attn_0_3_output):
        key = (num_attn_1_7_output, num_attn_0_3_output)
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
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 10),
            (2, 11),
            (2, 12),
            (2, 13),
            (2, 14),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (3, 6),
            (3, 7),
            (3, 8),
            (3, 9),
            (3, 10),
            (3, 11),
            (3, 12),
            (3, 13),
            (3, 14),
            (3, 15),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 7),
            (4, 8),
            (4, 9),
            (4, 10),
            (4, 11),
            (4, 12),
            (4, 13),
            (4, 14),
            (4, 15),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (5, 6),
            (5, 7),
            (5, 8),
            (5, 9),
            (5, 10),
            (5, 11),
            (5, 12),
            (5, 13),
            (5, 14),
            (5, 15),
            (5, 16),
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
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
            (7, 4),
            (7, 5),
            (7, 6),
            (7, 7),
            (7, 8),
            (7, 9),
            (7, 10),
            (7, 11),
            (7, 12),
            (7, 13),
            (7, 14),
            (7, 15),
            (7, 16),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
            (8, 6),
            (8, 7),
            (8, 8),
            (8, 9),
            (8, 10),
            (8, 11),
            (8, 12),
            (8, 13),
            (8, 14),
            (8, 15),
            (8, 16),
            (8, 17),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (9, 4),
            (9, 5),
            (9, 6),
            (9, 7),
            (9, 8),
            (9, 9),
            (9, 10),
            (9, 11),
            (9, 12),
            (9, 13),
            (9, 14),
            (9, 15),
            (9, 16),
            (9, 17),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (10, 5),
            (10, 6),
            (10, 7),
            (10, 8),
            (10, 9),
            (10, 10),
            (10, 11),
            (10, 12),
            (10, 13),
            (10, 14),
            (10, 15),
            (10, 16),
            (10, 17),
            (10, 18),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (11, 4),
            (11, 5),
            (11, 6),
            (11, 7),
            (11, 8),
            (11, 9),
            (11, 10),
            (11, 11),
            (11, 12),
            (11, 13),
            (11, 14),
            (11, 15),
            (11, 16),
            (11, 17),
            (11, 18),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
            (12, 4),
            (12, 5),
            (12, 6),
            (12, 7),
            (12, 8),
            (12, 9),
            (12, 10),
            (12, 11),
            (12, 12),
            (12, 13),
            (12, 14),
            (12, 15),
            (12, 16),
            (12, 17),
            (12, 18),
            (12, 19),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (13, 4),
            (13, 5),
            (13, 6),
            (13, 7),
            (13, 8),
            (13, 9),
            (13, 10),
            (13, 11),
            (13, 12),
            (13, 13),
            (13, 14),
            (13, 15),
            (13, 16),
            (13, 17),
            (13, 18),
            (13, 19),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (14, 4),
            (14, 5),
            (14, 6),
            (14, 7),
            (14, 8),
            (14, 9),
            (14, 10),
            (14, 11),
            (14, 12),
            (14, 13),
            (14, 14),
            (14, 15),
            (14, 16),
            (14, 17),
            (14, 18),
            (14, 19),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (15, 4),
            (15, 5),
            (15, 6),
            (15, 7),
            (15, 8),
            (15, 9),
            (15, 10),
            (15, 11),
            (15, 12),
            (15, 13),
            (15, 14),
            (15, 15),
            (15, 16),
            (15, 17),
            (15, 18),
            (15, 19),
            (15, 20),
            (16, 0),
            (16, 1),
            (16, 2),
            (16, 3),
            (16, 4),
            (16, 5),
            (16, 6),
            (16, 7),
            (16, 8),
            (16, 9),
            (16, 10),
            (16, 11),
            (16, 12),
            (16, 13),
            (16, 14),
            (16, 15),
            (16, 16),
            (16, 17),
            (16, 18),
            (16, 19),
            (16, 20),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (17, 4),
            (17, 5),
            (17, 6),
            (17, 7),
            (17, 8),
            (17, 9),
            (17, 10),
            (17, 11),
            (17, 12),
            (17, 13),
            (17, 14),
            (17, 15),
            (17, 16),
            (17, 17),
            (17, 18),
            (17, 19),
            (17, 20),
            (17, 21),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (18, 4),
            (18, 5),
            (18, 6),
            (18, 7),
            (18, 8),
            (18, 9),
            (18, 10),
            (18, 11),
            (18, 12),
            (18, 13),
            (18, 14),
            (18, 15),
            (18, 16),
            (18, 17),
            (18, 18),
            (18, 19),
            (18, 20),
            (18, 21),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (19, 4),
            (19, 5),
            (19, 6),
            (19, 7),
            (19, 8),
            (19, 9),
            (19, 10),
            (19, 11),
            (19, 12),
            (19, 13),
            (19, 14),
            (19, 15),
            (19, 16),
            (19, 17),
            (19, 18),
            (19, 19),
            (19, 20),
            (19, 21),
            (19, 22),
            (20, 0),
            (20, 1),
            (20, 2),
            (20, 3),
            (20, 4),
            (20, 5),
            (20, 6),
            (20, 7),
            (20, 8),
            (20, 9),
            (20, 10),
            (20, 11),
            (20, 12),
            (20, 13),
            (20, 14),
            (20, 15),
            (20, 16),
            (20, 17),
            (20, 18),
            (20, 19),
            (20, 20),
            (20, 21),
            (20, 22),
            (21, 0),
            (21, 1),
            (21, 2),
            (21, 3),
            (21, 4),
            (21, 5),
            (21, 6),
            (21, 7),
            (21, 8),
            (21, 9),
            (21, 10),
            (21, 11),
            (21, 12),
            (21, 13),
            (21, 14),
            (21, 15),
            (21, 16),
            (21, 17),
            (21, 18),
            (21, 19),
            (21, 20),
            (21, 21),
            (21, 22),
            (22, 0),
            (22, 1),
            (22, 2),
            (22, 3),
            (22, 4),
            (22, 5),
            (22, 6),
            (22, 7),
            (22, 8),
            (22, 9),
            (22, 10),
            (22, 11),
            (22, 12),
            (22, 13),
            (22, 14),
            (22, 15),
            (22, 16),
            (22, 17),
            (22, 18),
            (22, 19),
            (22, 20),
            (22, 21),
            (22, 22),
            (22, 23),
            (23, 0),
            (23, 1),
            (23, 2),
            (23, 3),
            (23, 4),
            (23, 5),
            (23, 6),
            (23, 7),
            (23, 8),
            (23, 9),
            (23, 10),
            (23, 11),
            (23, 12),
            (23, 13),
            (23, 14),
            (23, 15),
            (23, 16),
            (23, 17),
            (23, 18),
            (23, 19),
            (23, 20),
            (23, 21),
            (23, 22),
            (23, 23),
            (24, 0),
            (24, 1),
            (24, 2),
            (24, 3),
            (24, 4),
            (24, 5),
            (24, 6),
            (24, 7),
            (24, 8),
            (24, 9),
            (24, 10),
            (24, 11),
            (24, 12),
            (24, 13),
            (24, 14),
            (24, 15),
            (24, 16),
            (24, 17),
            (24, 18),
            (24, 19),
            (24, 20),
            (24, 21),
            (24, 22),
            (24, 23),
            (24, 24),
            (25, 0),
            (25, 1),
            (25, 2),
            (25, 3),
            (25, 4),
            (25, 5),
            (25, 6),
            (25, 7),
            (25, 8),
            (25, 9),
            (25, 10),
            (25, 11),
            (25, 12),
            (25, 13),
            (25, 14),
            (25, 15),
            (25, 16),
            (25, 17),
            (25, 18),
            (25, 19),
            (25, 20),
            (25, 21),
            (25, 22),
            (25, 23),
            (25, 24),
            (26, 0),
            (26, 1),
            (26, 2),
            (26, 3),
            (26, 4),
            (26, 5),
            (26, 6),
            (26, 7),
            (26, 8),
            (26, 9),
            (26, 10),
            (26, 11),
            (26, 12),
            (26, 13),
            (26, 14),
            (26, 15),
            (26, 16),
            (26, 17),
            (26, 18),
            (26, 19),
            (26, 20),
            (26, 21),
            (26, 22),
            (26, 23),
            (26, 24),
            (26, 25),
            (27, 0),
            (27, 1),
            (27, 2),
            (27, 3),
            (27, 4),
            (27, 5),
            (27, 6),
            (27, 7),
            (27, 8),
            (27, 9),
            (27, 10),
            (27, 11),
            (27, 12),
            (27, 13),
            (27, 14),
            (27, 15),
            (27, 16),
            (27, 17),
            (27, 18),
            (27, 19),
            (27, 20),
            (27, 21),
            (27, 22),
            (27, 23),
            (27, 24),
            (27, 25),
            (28, 0),
            (28, 1),
            (28, 2),
            (28, 3),
            (28, 4),
            (28, 5),
            (28, 6),
            (28, 7),
            (28, 8),
            (28, 9),
            (28, 10),
            (28, 11),
            (28, 12),
            (28, 13),
            (28, 14),
            (28, 15),
            (28, 16),
            (28, 17),
            (28, 18),
            (28, 19),
            (28, 20),
            (28, 21),
            (28, 22),
            (28, 23),
            (28, 24),
            (28, 25),
            (29, 0),
            (29, 1),
            (29, 2),
            (29, 3),
            (29, 4),
            (29, 5),
            (29, 6),
            (29, 7),
            (29, 8),
            (29, 9),
            (29, 10),
            (29, 11),
            (29, 12),
            (29, 13),
            (29, 14),
            (29, 15),
            (29, 16),
            (29, 17),
            (29, 18),
            (29, 19),
            (29, 20),
            (29, 21),
            (29, 22),
            (29, 23),
            (29, 24),
            (29, 25),
            (29, 26),
            (30, 0),
            (30, 1),
            (30, 2),
            (30, 3),
            (30, 4),
            (30, 5),
            (30, 6),
            (30, 7),
            (30, 8),
            (30, 9),
            (30, 10),
            (30, 11),
            (30, 12),
            (30, 13),
            (30, 14),
            (30, 15),
            (30, 16),
            (30, 17),
            (30, 18),
            (30, 19),
            (30, 20),
            (30, 21),
            (30, 22),
            (30, 23),
            (30, 24),
            (30, 25),
            (30, 26),
            (31, 0),
            (31, 1),
            (31, 2),
            (31, 3),
            (31, 4),
            (31, 5),
            (31, 6),
            (31, 7),
            (31, 8),
            (31, 9),
            (31, 10),
            (31, 11),
            (31, 12),
            (31, 13),
            (31, 14),
            (31, 15),
            (31, 16),
            (31, 17),
            (31, 18),
            (31, 19),
            (31, 20),
            (31, 21),
            (31, 22),
            (31, 23),
            (31, 24),
            (31, 25),
            (31, 26),
            (31, 27),
            (32, 0),
            (32, 1),
            (32, 2),
            (32, 3),
            (32, 4),
            (32, 5),
            (32, 6),
            (32, 7),
            (32, 8),
            (32, 9),
            (32, 10),
            (32, 11),
            (32, 12),
            (32, 13),
            (32, 14),
            (32, 15),
            (32, 16),
            (32, 17),
            (32, 18),
            (32, 19),
            (32, 20),
            (32, 21),
            (32, 22),
            (32, 23),
            (32, 24),
            (32, 25),
            (32, 26),
            (32, 27),
            (33, 0),
            (33, 1),
            (33, 2),
            (33, 3),
            (33, 4),
            (33, 5),
            (33, 6),
            (33, 7),
            (33, 8),
            (33, 9),
            (33, 10),
            (33, 11),
            (33, 12),
            (33, 13),
            (33, 14),
            (33, 15),
            (33, 16),
            (33, 17),
            (33, 18),
            (33, 19),
            (33, 20),
            (33, 21),
            (33, 22),
            (33, 23),
            (33, 24),
            (33, 25),
            (33, 26),
            (33, 27),
            (33, 28),
            (34, 0),
            (34, 1),
            (34, 2),
            (34, 3),
            (34, 4),
            (34, 5),
            (34, 6),
            (34, 7),
            (34, 8),
            (34, 9),
            (34, 10),
            (34, 11),
            (34, 12),
            (34, 13),
            (34, 14),
            (34, 15),
            (34, 16),
            (34, 17),
            (34, 18),
            (34, 19),
            (34, 20),
            (34, 21),
            (34, 22),
            (34, 23),
            (34, 24),
            (34, 25),
            (34, 26),
            (34, 27),
            (34, 28),
            (35, 0),
            (35, 1),
            (35, 2),
            (35, 3),
            (35, 4),
            (35, 5),
            (35, 6),
            (35, 7),
            (35, 8),
            (35, 9),
            (35, 10),
            (35, 11),
            (35, 12),
            (35, 13),
            (35, 14),
            (35, 15),
            (35, 16),
            (35, 17),
            (35, 18),
            (35, 19),
            (35, 20),
            (35, 21),
            (35, 22),
            (35, 23),
            (35, 24),
            (35, 25),
            (35, 26),
            (35, 27),
            (35, 28),
            (36, 0),
            (36, 1),
            (36, 2),
            (36, 3),
            (36, 4),
            (36, 5),
            (36, 6),
            (36, 7),
            (36, 8),
            (36, 9),
            (36, 10),
            (36, 11),
            (36, 12),
            (36, 13),
            (36, 14),
            (36, 15),
            (36, 16),
            (36, 17),
            (36, 18),
            (36, 19),
            (36, 20),
            (36, 21),
            (36, 22),
            (36, 23),
            (36, 24),
            (36, 25),
            (36, 26),
            (36, 27),
            (36, 28),
            (36, 29),
            (37, 0),
            (37, 1),
            (37, 2),
            (37, 3),
            (37, 4),
            (37, 5),
            (37, 6),
            (37, 7),
            (37, 8),
            (37, 9),
            (37, 10),
            (37, 11),
            (37, 12),
            (37, 13),
            (37, 14),
            (37, 15),
            (37, 16),
            (37, 17),
            (37, 18),
            (37, 19),
            (37, 20),
            (37, 21),
            (37, 22),
            (37, 23),
            (37, 24),
            (37, 25),
            (37, 26),
            (37, 27),
            (37, 28),
            (37, 29),
            (38, 0),
            (38, 1),
            (38, 2),
            (38, 3),
            (38, 4),
            (38, 5),
            (38, 6),
            (38, 7),
            (38, 8),
            (38, 9),
            (38, 10),
            (38, 11),
            (38, 12),
            (38, 13),
            (38, 14),
            (38, 15),
            (38, 16),
            (38, 17),
            (38, 18),
            (38, 19),
            (38, 20),
            (38, 21),
            (38, 22),
            (38, 23),
            (38, 24),
            (38, 25),
            (38, 26),
            (38, 27),
            (38, 28),
            (38, 29),
            (38, 30),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
            (39, 4),
            (39, 5),
            (39, 6),
            (39, 7),
            (39, 8),
            (39, 9),
            (39, 10),
            (39, 11),
            (39, 12),
            (39, 13),
            (39, 14),
            (39, 15),
            (39, 16),
            (39, 17),
            (39, 18),
            (39, 19),
            (39, 20),
            (39, 21),
            (39, 22),
            (39, 23),
            (39, 24),
            (39, 25),
            (39, 26),
            (39, 27),
            (39, 28),
            (39, 29),
            (39, 30),
            (40, 0),
            (40, 1),
            (40, 2),
            (40, 3),
            (40, 4),
            (40, 5),
            (40, 6),
            (40, 7),
            (40, 8),
            (40, 9),
            (40, 10),
            (40, 11),
            (40, 12),
            (40, 13),
            (40, 14),
            (40, 15),
            (40, 16),
            (40, 17),
            (40, 18),
            (40, 19),
            (40, 20),
            (40, 21),
            (40, 22),
            (40, 23),
            (40, 24),
            (40, 25),
            (40, 26),
            (40, 27),
            (40, 28),
            (40, 29),
            (40, 30),
            (40, 31),
            (41, 0),
            (41, 1),
            (41, 2),
            (41, 3),
            (41, 4),
            (41, 5),
            (41, 6),
            (41, 7),
            (41, 8),
            (41, 9),
            (41, 10),
            (41, 11),
            (41, 12),
            (41, 13),
            (41, 14),
            (41, 15),
            (41, 16),
            (41, 17),
            (41, 18),
            (41, 19),
            (41, 20),
            (41, 21),
            (41, 22),
            (41, 23),
            (41, 24),
            (41, 25),
            (41, 26),
            (41, 27),
            (41, 28),
            (41, 29),
            (41, 30),
            (41, 31),
            (42, 0),
            (42, 1),
            (42, 2),
            (42, 3),
            (42, 4),
            (42, 5),
            (42, 6),
            (42, 7),
            (42, 8),
            (42, 9),
            (42, 10),
            (42, 11),
            (42, 12),
            (42, 13),
            (42, 14),
            (42, 15),
            (42, 16),
            (42, 17),
            (42, 18),
            (42, 19),
            (42, 20),
            (42, 21),
            (42, 22),
            (42, 23),
            (42, 24),
            (42, 25),
            (42, 26),
            (42, 27),
            (42, 28),
            (42, 29),
            (42, 30),
            (42, 31),
            (43, 0),
            (43, 1),
            (43, 2),
            (43, 3),
            (43, 4),
            (43, 5),
            (43, 6),
            (43, 7),
            (43, 8),
            (43, 9),
            (43, 10),
            (43, 11),
            (43, 12),
            (43, 13),
            (43, 14),
            (43, 15),
            (43, 16),
            (43, 17),
            (43, 18),
            (43, 19),
            (43, 20),
            (43, 21),
            (43, 22),
            (43, 23),
            (43, 24),
            (43, 25),
            (43, 26),
            (43, 27),
            (43, 28),
            (43, 29),
            (43, 30),
            (43, 31),
            (43, 32),
            (44, 0),
            (44, 1),
            (44, 2),
            (44, 3),
            (44, 4),
            (44, 5),
            (44, 6),
            (44, 7),
            (44, 8),
            (44, 9),
            (44, 10),
            (44, 11),
            (44, 12),
            (44, 13),
            (44, 14),
            (44, 15),
            (44, 16),
            (44, 17),
            (44, 18),
            (44, 19),
            (44, 20),
            (44, 21),
            (44, 22),
            (44, 23),
            (44, 24),
            (44, 25),
            (44, 26),
            (44, 27),
            (44, 28),
            (44, 29),
            (44, 30),
            (44, 31),
            (44, 32),
            (45, 0),
            (45, 1),
            (45, 2),
            (45, 3),
            (45, 4),
            (45, 5),
            (45, 6),
            (45, 7),
            (45, 8),
            (45, 9),
            (45, 10),
            (45, 11),
            (45, 12),
            (45, 13),
            (45, 14),
            (45, 15),
            (45, 16),
            (45, 17),
            (45, 18),
            (45, 19),
            (45, 20),
            (45, 21),
            (45, 22),
            (45, 23),
            (45, 24),
            (45, 25),
            (45, 26),
            (45, 27),
            (45, 28),
            (45, 29),
            (45, 30),
            (45, 31),
            (45, 32),
            (45, 33),
            (46, 0),
            (46, 1),
            (46, 2),
            (46, 3),
            (46, 4),
            (46, 5),
            (46, 6),
            (46, 7),
            (46, 8),
            (46, 9),
            (46, 10),
            (46, 11),
            (46, 12),
            (46, 13),
            (46, 14),
            (46, 15),
            (46, 16),
            (46, 17),
            (46, 18),
            (46, 19),
            (46, 20),
            (46, 21),
            (46, 22),
            (46, 23),
            (46, 24),
            (46, 25),
            (46, 26),
            (46, 27),
            (46, 28),
            (46, 29),
            (46, 30),
            (46, 31),
            (46, 32),
            (46, 33),
            (47, 0),
            (47, 1),
            (47, 2),
            (47, 3),
            (47, 4),
            (47, 5),
            (47, 6),
            (47, 7),
            (47, 8),
            (47, 9),
            (47, 10),
            (47, 11),
            (47, 12),
            (47, 13),
            (47, 14),
            (47, 15),
            (47, 16),
            (47, 17),
            (47, 18),
            (47, 19),
            (47, 20),
            (47, 21),
            (47, 22),
            (47, 23),
            (47, 24),
            (47, 25),
            (47, 26),
            (47, 27),
            (47, 28),
            (47, 29),
            (47, 30),
            (47, 31),
            (47, 32),
            (47, 33),
            (47, 34),
            (48, 0),
            (48, 1),
            (48, 2),
            (48, 3),
            (48, 4),
            (48, 5),
            (48, 6),
            (48, 7),
            (48, 8),
            (48, 9),
            (48, 10),
            (48, 11),
            (48, 12),
            (48, 13),
            (48, 14),
            (48, 15),
            (48, 16),
            (48, 17),
            (48, 18),
            (48, 19),
            (48, 20),
            (48, 21),
            (48, 22),
            (48, 23),
            (48, 24),
            (48, 25),
            (48, 26),
            (48, 27),
            (48, 28),
            (48, 29),
            (48, 30),
            (48, 31),
            (48, 32),
            (48, 33),
            (48, 34),
            (49, 0),
            (49, 1),
            (49, 2),
            (49, 3),
            (49, 4),
            (49, 5),
            (49, 6),
            (49, 7),
            (49, 8),
            (49, 9),
            (49, 10),
            (49, 11),
            (49, 12),
            (49, 13),
            (49, 14),
            (49, 15),
            (49, 16),
            (49, 17),
            (49, 18),
            (49, 19),
            (49, 20),
            (49, 21),
            (49, 22),
            (49, 23),
            (49, 24),
            (49, 25),
            (49, 26),
            (49, 27),
            (49, 28),
            (49, 29),
            (49, 30),
            (49, 31),
            (49, 32),
            (49, 33),
            (49, 34),
            (50, 0),
            (50, 1),
            (50, 2),
            (50, 3),
            (50, 4),
            (50, 5),
            (50, 6),
            (50, 7),
            (50, 8),
            (50, 9),
            (50, 10),
            (50, 11),
            (50, 12),
            (50, 13),
            (50, 14),
            (50, 15),
            (50, 16),
            (50, 17),
            (50, 18),
            (50, 19),
            (50, 20),
            (50, 21),
            (50, 22),
            (50, 23),
            (50, 24),
            (50, 25),
            (50, 26),
            (50, 27),
            (50, 28),
            (50, 29),
            (50, 30),
            (50, 31),
            (50, 32),
            (50, 33),
            (50, 34),
            (50, 35),
            (51, 0),
            (51, 1),
            (51, 2),
            (51, 3),
            (51, 4),
            (51, 5),
            (51, 6),
            (51, 7),
            (51, 8),
            (51, 9),
            (51, 10),
            (51, 11),
            (51, 12),
            (51, 13),
            (51, 14),
            (51, 15),
            (51, 16),
            (51, 17),
            (51, 18),
            (51, 19),
            (51, 20),
            (51, 21),
            (51, 22),
            (51, 23),
            (51, 24),
            (51, 25),
            (51, 26),
            (51, 27),
            (51, 28),
            (51, 29),
            (51, 30),
            (51, 31),
            (51, 32),
            (51, 33),
            (51, 34),
            (51, 35),
            (52, 0),
            (52, 1),
            (52, 2),
            (52, 3),
            (52, 4),
            (52, 5),
            (52, 6),
            (52, 7),
            (52, 8),
            (52, 9),
            (52, 10),
            (52, 11),
            (52, 12),
            (52, 13),
            (52, 14),
            (52, 15),
            (52, 16),
            (52, 17),
            (52, 18),
            (52, 19),
            (52, 20),
            (52, 21),
            (52, 22),
            (52, 23),
            (52, 24),
            (52, 25),
            (52, 26),
            (52, 27),
            (52, 28),
            (52, 29),
            (52, 30),
            (52, 31),
            (52, 32),
            (52, 33),
            (52, 34),
            (52, 35),
            (52, 36),
            (53, 0),
            (53, 1),
            (53, 2),
            (53, 3),
            (53, 4),
            (53, 5),
            (53, 6),
            (53, 7),
            (53, 8),
            (53, 9),
            (53, 10),
            (53, 11),
            (53, 12),
            (53, 13),
            (53, 14),
            (53, 15),
            (53, 16),
            (53, 17),
            (53, 18),
            (53, 19),
            (53, 20),
            (53, 21),
            (53, 22),
            (53, 23),
            (53, 24),
            (53, 25),
            (53, 26),
            (53, 27),
            (53, 28),
            (53, 29),
            (53, 30),
            (53, 31),
            (53, 32),
            (53, 33),
            (53, 34),
            (53, 35),
            (53, 36),
            (54, 0),
            (54, 1),
            (54, 2),
            (54, 3),
            (54, 4),
            (54, 5),
            (54, 6),
            (54, 7),
            (54, 8),
            (54, 9),
            (54, 10),
            (54, 11),
            (54, 12),
            (54, 13),
            (54, 14),
            (54, 15),
            (54, 16),
            (54, 17),
            (54, 18),
            (54, 19),
            (54, 20),
            (54, 21),
            (54, 22),
            (54, 23),
            (54, 24),
            (54, 25),
            (54, 26),
            (54, 27),
            (54, 28),
            (54, 29),
            (54, 30),
            (54, 31),
            (54, 32),
            (54, 33),
            (54, 34),
            (54, 35),
            (54, 36),
            (54, 37),
            (55, 0),
            (55, 1),
            (55, 2),
            (55, 3),
            (55, 4),
            (55, 5),
            (55, 6),
            (55, 7),
            (55, 8),
            (55, 9),
            (55, 10),
            (55, 11),
            (55, 12),
            (55, 13),
            (55, 14),
            (55, 15),
            (55, 16),
            (55, 17),
            (55, 18),
            (55, 19),
            (55, 20),
            (55, 21),
            (55, 22),
            (55, 23),
            (55, 24),
            (55, 25),
            (55, 26),
            (55, 27),
            (55, 28),
            (55, 29),
            (55, 30),
            (55, 31),
            (55, 32),
            (55, 33),
            (55, 34),
            (55, 35),
            (55, 36),
            (55, 37),
            (56, 0),
            (56, 1),
            (56, 2),
            (56, 3),
            (56, 4),
            (56, 5),
            (56, 6),
            (56, 7),
            (56, 8),
            (56, 9),
            (56, 10),
            (56, 11),
            (56, 12),
            (56, 13),
            (56, 14),
            (56, 15),
            (56, 16),
            (56, 17),
            (56, 18),
            (56, 19),
            (56, 20),
            (56, 21),
            (56, 22),
            (56, 23),
            (56, 24),
            (56, 25),
            (56, 26),
            (56, 27),
            (56, 28),
            (56, 29),
            (56, 30),
            (56, 31),
            (56, 32),
            (56, 33),
            (56, 34),
            (56, 35),
            (56, 36),
            (56, 37),
            (57, 0),
            (57, 1),
            (57, 2),
            (57, 3),
            (57, 4),
            (57, 5),
            (57, 6),
            (57, 7),
            (57, 8),
            (57, 9),
            (57, 10),
            (57, 11),
            (57, 12),
            (57, 13),
            (57, 14),
            (57, 15),
            (57, 16),
            (57, 17),
            (57, 18),
            (57, 19),
            (57, 20),
            (57, 21),
            (57, 22),
            (57, 23),
            (57, 24),
            (57, 25),
            (57, 26),
            (57, 27),
            (57, 28),
            (57, 29),
            (57, 30),
            (57, 31),
            (57, 32),
            (57, 33),
            (57, 34),
            (57, 35),
            (57, 36),
            (57, 37),
            (57, 38),
            (58, 0),
            (58, 1),
            (58, 2),
            (58, 3),
            (58, 4),
            (58, 5),
            (58, 6),
            (58, 7),
            (58, 8),
            (58, 9),
            (58, 10),
            (58, 11),
            (58, 12),
            (58, 13),
            (58, 14),
            (58, 15),
            (58, 16),
            (58, 17),
            (58, 18),
            (58, 19),
            (58, 20),
            (58, 21),
            (58, 22),
            (58, 23),
            (58, 24),
            (58, 25),
            (58, 26),
            (58, 27),
            (58, 28),
            (58, 29),
            (58, 30),
            (58, 31),
            (58, 32),
            (58, 33),
            (58, 34),
            (58, 35),
            (58, 36),
            (58, 37),
            (58, 38),
            (59, 0),
            (59, 1),
            (59, 2),
            (59, 3),
            (59, 4),
            (59, 5),
            (59, 6),
            (59, 7),
            (59, 8),
            (59, 9),
            (59, 10),
            (59, 11),
            (59, 12),
            (59, 13),
            (59, 14),
            (59, 15),
            (59, 16),
            (59, 17),
            (59, 18),
            (59, 19),
            (59, 20),
            (59, 21),
            (59, 22),
            (59, 23),
            (59, 24),
            (59, 25),
            (59, 26),
            (59, 27),
            (59, 28),
            (59, 29),
            (59, 30),
            (59, 31),
            (59, 32),
            (59, 33),
            (59, 34),
            (59, 35),
            (59, 36),
            (59, 37),
            (59, 38),
            (59, 39),
            (60, 0),
            (60, 1),
            (60, 2),
            (60, 3),
            (60, 4),
            (60, 5),
            (60, 6),
            (60, 7),
            (60, 8),
            (60, 9),
            (60, 10),
            (60, 11),
            (60, 12),
            (60, 13),
            (60, 14),
            (60, 15),
            (60, 16),
            (60, 17),
            (60, 18),
            (60, 19),
            (60, 20),
            (60, 21),
            (60, 22),
            (60, 23),
            (60, 24),
            (60, 25),
            (60, 26),
            (60, 27),
            (60, 28),
            (60, 29),
            (60, 30),
            (60, 31),
            (60, 32),
            (60, 33),
            (60, 34),
            (60, 35),
            (60, 36),
            (60, 37),
            (60, 38),
            (60, 39),
            (61, 0),
            (61, 1),
            (61, 2),
            (61, 3),
            (61, 4),
            (61, 5),
            (61, 6),
            (61, 7),
            (61, 8),
            (61, 9),
            (61, 10),
            (61, 11),
            (61, 12),
            (61, 13),
            (61, 14),
            (61, 15),
            (61, 16),
            (61, 17),
            (61, 18),
            (61, 19),
            (61, 20),
            (61, 21),
            (61, 22),
            (61, 23),
            (61, 24),
            (61, 25),
            (61, 26),
            (61, 27),
            (61, 28),
            (61, 29),
            (61, 30),
            (61, 31),
            (61, 32),
            (61, 33),
            (61, 34),
            (61, 35),
            (61, 36),
            (61, 37),
            (61, 38),
            (61, 39),
            (61, 40),
            (62, 0),
            (62, 1),
            (62, 2),
            (62, 3),
            (62, 4),
            (62, 5),
            (62, 6),
            (62, 7),
            (62, 8),
            (62, 9),
            (62, 10),
            (62, 11),
            (62, 12),
            (62, 13),
            (62, 14),
            (62, 15),
            (62, 16),
            (62, 17),
            (62, 18),
            (62, 19),
            (62, 20),
            (62, 21),
            (62, 22),
            (62, 23),
            (62, 24),
            (62, 25),
            (62, 26),
            (62, 27),
            (62, 28),
            (62, 29),
            (62, 30),
            (62, 31),
            (62, 32),
            (62, 33),
            (62, 34),
            (62, 35),
            (62, 36),
            (62, 37),
            (62, 38),
            (62, 39),
            (62, 40),
            (63, 0),
            (63, 1),
            (63, 2),
            (63, 3),
            (63, 4),
            (63, 5),
            (63, 6),
            (63, 7),
            (63, 8),
            (63, 9),
            (63, 10),
            (63, 11),
            (63, 12),
            (63, 13),
            (63, 14),
            (63, 15),
            (63, 16),
            (63, 17),
            (63, 18),
            (63, 19),
            (63, 20),
            (63, 21),
            (63, 22),
            (63, 23),
            (63, 24),
            (63, 25),
            (63, 26),
            (63, 27),
            (63, 28),
            (63, 29),
            (63, 30),
            (63, 31),
            (63, 32),
            (63, 33),
            (63, 34),
            (63, 35),
            (63, 36),
            (63, 37),
            (63, 38),
            (63, 39),
            (63, 40),
            (64, 0),
            (64, 1),
            (64, 2),
            (64, 3),
            (64, 4),
            (64, 5),
            (64, 6),
            (64, 7),
            (64, 8),
            (64, 9),
            (64, 10),
            (64, 11),
            (64, 12),
            (64, 13),
            (64, 14),
            (64, 15),
            (64, 16),
            (64, 17),
            (64, 18),
            (64, 19),
            (64, 20),
            (64, 21),
            (64, 22),
            (64, 23),
            (64, 24),
            (64, 25),
            (64, 26),
            (64, 27),
            (64, 28),
            (64, 29),
            (64, 30),
            (64, 31),
            (64, 32),
            (64, 33),
            (64, 34),
            (64, 35),
            (64, 36),
            (64, 37),
            (64, 38),
            (64, 39),
            (64, 40),
            (64, 41),
            (65, 0),
            (65, 1),
            (65, 2),
            (65, 3),
            (65, 4),
            (65, 5),
            (65, 6),
            (65, 7),
            (65, 8),
            (65, 9),
            (65, 10),
            (65, 11),
            (65, 12),
            (65, 13),
            (65, 14),
            (65, 15),
            (65, 16),
            (65, 17),
            (65, 18),
            (65, 19),
            (65, 20),
            (65, 21),
            (65, 22),
            (65, 23),
            (65, 24),
            (65, 25),
            (65, 26),
            (65, 27),
            (65, 28),
            (65, 29),
            (65, 30),
            (65, 31),
            (65, 32),
            (65, 33),
            (65, 34),
            (65, 35),
            (65, 36),
            (65, 37),
            (65, 38),
            (65, 39),
            (65, 40),
            (65, 41),
            (66, 0),
            (66, 1),
            (66, 2),
            (66, 3),
            (66, 4),
            (66, 5),
            (66, 6),
            (66, 7),
            (66, 8),
            (66, 9),
            (66, 10),
            (66, 11),
            (66, 12),
            (66, 13),
            (66, 14),
            (66, 15),
            (66, 16),
            (66, 17),
            (66, 18),
            (66, 19),
            (66, 20),
            (66, 21),
            (66, 22),
            (66, 23),
            (66, 24),
            (66, 25),
            (66, 26),
            (66, 27),
            (66, 28),
            (66, 29),
            (66, 30),
            (66, 31),
            (66, 32),
            (66, 33),
            (66, 34),
            (66, 35),
            (66, 36),
            (66, 37),
            (66, 38),
            (66, 39),
            (66, 40),
            (66, 41),
            (66, 42),
            (67, 0),
            (67, 1),
            (67, 2),
            (67, 3),
            (67, 4),
            (67, 5),
            (67, 6),
            (67, 7),
            (67, 8),
            (67, 9),
            (67, 10),
            (67, 11),
            (67, 12),
            (67, 13),
            (67, 14),
            (67, 15),
            (67, 16),
            (67, 17),
            (67, 18),
            (67, 19),
            (67, 20),
            (67, 21),
            (67, 22),
            (67, 23),
            (67, 24),
            (67, 25),
            (67, 26),
            (67, 27),
            (67, 28),
            (67, 29),
            (67, 30),
            (67, 31),
            (67, 32),
            (67, 33),
            (67, 34),
            (67, 35),
            (67, 36),
            (67, 37),
            (67, 38),
            (67, 39),
            (67, 40),
            (67, 41),
            (67, 42),
            (68, 0),
            (68, 1),
            (68, 2),
            (68, 3),
            (68, 4),
            (68, 5),
            (68, 6),
            (68, 7),
            (68, 8),
            (68, 9),
            (68, 10),
            (68, 11),
            (68, 12),
            (68, 13),
            (68, 14),
            (68, 15),
            (68, 16),
            (68, 17),
            (68, 18),
            (68, 19),
            (68, 20),
            (68, 21),
            (68, 22),
            (68, 23),
            (68, 24),
            (68, 25),
            (68, 26),
            (68, 27),
            (68, 28),
            (68, 29),
            (68, 30),
            (68, 31),
            (68, 32),
            (68, 33),
            (68, 34),
            (68, 35),
            (68, 36),
            (68, 37),
            (68, 38),
            (68, 39),
            (68, 40),
            (68, 41),
            (68, 42),
            (68, 43),
            (69, 0),
            (69, 1),
            (69, 2),
            (69, 3),
            (69, 4),
            (69, 5),
            (69, 6),
            (69, 7),
            (69, 8),
            (69, 9),
            (69, 10),
            (69, 11),
            (69, 12),
            (69, 13),
            (69, 14),
            (69, 15),
            (69, 16),
            (69, 17),
            (69, 18),
            (69, 19),
            (69, 20),
            (69, 21),
            (69, 22),
            (69, 23),
            (69, 24),
            (69, 25),
            (69, 26),
            (69, 27),
            (69, 28),
            (69, 29),
            (69, 30),
            (69, 31),
            (69, 32),
            (69, 33),
            (69, 34),
            (69, 35),
            (69, 36),
            (69, 37),
            (69, 38),
            (69, 39),
            (69, 40),
            (69, 41),
            (69, 42),
            (69, 43),
            (70, 0),
            (70, 1),
            (70, 2),
            (70, 3),
            (70, 4),
            (70, 5),
            (70, 6),
            (70, 7),
            (70, 8),
            (70, 9),
            (70, 10),
            (70, 11),
            (70, 12),
            (70, 13),
            (70, 14),
            (70, 15),
            (70, 16),
            (70, 17),
            (70, 18),
            (70, 19),
            (70, 20),
            (70, 21),
            (70, 22),
            (70, 23),
            (70, 24),
            (70, 25),
            (70, 26),
            (70, 27),
            (70, 28),
            (70, 29),
            (70, 30),
            (70, 31),
            (70, 32),
            (70, 33),
            (70, 34),
            (70, 35),
            (70, 36),
            (70, 37),
            (70, 38),
            (70, 39),
            (70, 40),
            (70, 41),
            (70, 42),
            (70, 43),
            (71, 0),
            (71, 1),
            (71, 2),
            (71, 3),
            (71, 4),
            (71, 5),
            (71, 6),
            (71, 7),
            (71, 8),
            (71, 9),
            (71, 10),
            (71, 11),
            (71, 12),
            (71, 13),
            (71, 14),
            (71, 15),
            (71, 16),
            (71, 17),
            (71, 18),
            (71, 19),
            (71, 20),
            (71, 21),
            (71, 22),
            (71, 23),
            (71, 24),
            (71, 25),
            (71, 26),
            (71, 27),
            (71, 28),
            (71, 29),
            (71, 30),
            (71, 31),
            (71, 32),
            (71, 33),
            (71, 34),
            (71, 35),
            (71, 36),
            (71, 37),
            (71, 38),
            (71, 39),
            (71, 40),
            (71, 41),
            (71, 42),
            (71, 43),
            (71, 44),
            (72, 0),
            (72, 1),
            (72, 2),
            (72, 3),
            (72, 4),
            (72, 5),
            (72, 6),
            (72, 7),
            (72, 8),
            (72, 9),
            (72, 10),
            (72, 11),
            (72, 12),
            (72, 13),
            (72, 14),
            (72, 15),
            (72, 16),
            (72, 17),
            (72, 18),
            (72, 19),
            (72, 20),
            (72, 21),
            (72, 22),
            (72, 23),
            (72, 24),
            (72, 25),
            (72, 26),
            (72, 27),
            (72, 28),
            (72, 29),
            (72, 30),
            (72, 31),
            (72, 32),
            (72, 33),
            (72, 34),
            (72, 35),
            (72, 36),
            (72, 37),
            (72, 38),
            (72, 39),
            (72, 40),
            (72, 41),
            (72, 42),
            (72, 43),
            (72, 44),
            (73, 0),
            (73, 1),
            (73, 2),
            (73, 3),
            (73, 4),
            (73, 5),
            (73, 6),
            (73, 7),
            (73, 8),
            (73, 9),
            (73, 10),
            (73, 11),
            (73, 12),
            (73, 13),
            (73, 14),
            (73, 15),
            (73, 16),
            (73, 17),
            (73, 18),
            (73, 19),
            (73, 20),
            (73, 21),
            (73, 22),
            (73, 23),
            (73, 24),
            (73, 25),
            (73, 26),
            (73, 27),
            (73, 28),
            (73, 29),
            (73, 30),
            (73, 31),
            (73, 32),
            (73, 33),
            (73, 34),
            (73, 35),
            (73, 36),
            (73, 37),
            (73, 38),
            (73, 39),
            (73, 40),
            (73, 41),
            (73, 42),
            (73, 43),
            (73, 44),
            (73, 45),
            (74, 0),
            (74, 1),
            (74, 2),
            (74, 3),
            (74, 4),
            (74, 5),
            (74, 6),
            (74, 7),
            (74, 8),
            (74, 9),
            (74, 10),
            (74, 11),
            (74, 12),
            (74, 13),
            (74, 14),
            (74, 15),
            (74, 16),
            (74, 17),
            (74, 18),
            (74, 19),
            (74, 20),
            (74, 21),
            (74, 22),
            (74, 23),
            (74, 24),
            (74, 25),
            (74, 26),
            (74, 27),
            (74, 28),
            (74, 29),
            (74, 30),
            (74, 31),
            (74, 32),
            (74, 33),
            (74, 34),
            (74, 35),
            (74, 36),
            (74, 37),
            (74, 38),
            (74, 39),
            (74, 40),
            (74, 41),
            (74, 42),
            (74, 43),
            (74, 44),
            (74, 45),
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
            (75, 10),
            (75, 11),
            (75, 12),
            (75, 13),
            (75, 14),
            (75, 15),
            (75, 16),
            (75, 17),
            (75, 18),
            (75, 19),
            (75, 20),
            (75, 21),
            (75, 22),
            (75, 23),
            (75, 24),
            (75, 25),
            (75, 26),
            (75, 27),
            (75, 28),
            (75, 29),
            (75, 30),
            (75, 31),
            (75, 32),
            (75, 33),
            (75, 34),
            (75, 35),
            (75, 36),
            (75, 37),
            (75, 38),
            (75, 39),
            (75, 40),
            (75, 41),
            (75, 42),
            (75, 43),
            (75, 44),
            (75, 45),
            (75, 46),
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
            (76, 10),
            (76, 11),
            (76, 12),
            (76, 13),
            (76, 14),
            (76, 15),
            (76, 16),
            (76, 17),
            (76, 18),
            (76, 19),
            (76, 20),
            (76, 21),
            (76, 22),
            (76, 23),
            (76, 24),
            (76, 25),
            (76, 26),
            (76, 27),
            (76, 28),
            (76, 29),
            (76, 30),
            (76, 31),
            (76, 32),
            (76, 33),
            (76, 34),
            (76, 35),
            (76, 36),
            (76, 37),
            (76, 38),
            (76, 39),
            (76, 40),
            (76, 41),
            (76, 42),
            (76, 43),
            (76, 44),
            (76, 45),
            (76, 46),
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
            (77, 10),
            (77, 11),
            (77, 12),
            (77, 13),
            (77, 14),
            (77, 15),
            (77, 16),
            (77, 17),
            (77, 18),
            (77, 19),
            (77, 20),
            (77, 21),
            (77, 22),
            (77, 23),
            (77, 24),
            (77, 25),
            (77, 26),
            (77, 27),
            (77, 28),
            (77, 29),
            (77, 30),
            (77, 31),
            (77, 32),
            (77, 33),
            (77, 34),
            (77, 35),
            (77, 36),
            (77, 37),
            (77, 38),
            (77, 39),
            (77, 40),
            (77, 41),
            (77, 42),
            (77, 43),
            (77, 44),
            (77, 45),
            (77, 46),
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
            (78, 10),
            (78, 11),
            (78, 12),
            (78, 13),
            (78, 14),
            (78, 15),
            (78, 16),
            (78, 17),
            (78, 18),
            (78, 19),
            (78, 20),
            (78, 21),
            (78, 22),
            (78, 23),
            (78, 24),
            (78, 25),
            (78, 26),
            (78, 27),
            (78, 28),
            (78, 29),
            (78, 30),
            (78, 31),
            (78, 32),
            (78, 33),
            (78, 34),
            (78, 35),
            (78, 36),
            (78, 37),
            (78, 38),
            (78, 39),
            (78, 40),
            (78, 41),
            (78, 42),
            (78, 43),
            (78, 44),
            (78, 45),
            (78, 46),
            (78, 47),
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
            (79, 10),
            (79, 11),
            (79, 12),
            (79, 13),
            (79, 14),
            (79, 15),
            (79, 16),
            (79, 17),
            (79, 18),
            (79, 19),
            (79, 20),
            (79, 21),
            (79, 22),
            (79, 23),
            (79, 24),
            (79, 25),
            (79, 26),
            (79, 27),
            (79, 28),
            (79, 29),
            (79, 30),
            (79, 31),
            (79, 32),
            (79, 33),
            (79, 34),
            (79, 35),
            (79, 36),
            (79, 37),
            (79, 38),
            (79, 39),
            (79, 40),
            (79, 41),
            (79, 42),
            (79, 43),
            (79, 44),
            (79, 45),
            (79, 46),
            (79, 47),
        }:
            return 2
        return 16

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_0_output, mlp_1_0_output):
        if attn_0_0_output in {")", "(", "<s>"}:
            return mlp_1_0_output == 2

    attn_2_0_pattern = select_closest(mlp_1_0_outputs, attn_0_0_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_0_7_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(token, attn_0_5_output):
        if token in {")", "(", "<s>"}:
            return attn_0_5_output == ")"

    attn_2_1_pattern = select_closest(attn_0_5_outputs, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_4_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(token, attn_0_4_output):
        if token in {")", "("}:
            return attn_0_4_output == ")"
        elif token in {"<s>"}:
            return attn_0_4_output == ""

    attn_2_2_pattern = select_closest(attn_0_4_outputs, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_0_7_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(token, position):
        if token in {"(", "<s>"}:
            return position == 1
        elif token in {")"}:
            return position == 11

    attn_2_3_pattern = select_closest(positions, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_5_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_5_output, position):
        if attn_0_5_output in {"("}:
            return position == 3
        elif attn_0_5_output in {")"}:
            return position == 17
        elif attn_0_5_output in {"<s>"}:
            return position == 1

    attn_2_4_pattern = select_closest(positions, attn_0_5_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_4_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(token, position):
        if token in {")", "("}:
            return position == 3
        elif token in {"<s>"}:
            return position == 1

    attn_2_5_pattern = select_closest(positions, tokens, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_3_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(token, attn_0_5_output):
        if token in {")", "(", "<s>"}:
            return attn_0_5_output == ")"

    attn_2_6_pattern = select_closest(attn_0_5_outputs, tokens, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_4_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_0_5_output, position):
        if attn_0_5_output in {"("}:
            return position == 3
        elif attn_0_5_output in {")"}:
            return position == 11
        elif attn_0_5_output in {"<s>"}:
            return position == 2

    attn_2_7_pattern = select_closest(positions, attn_0_5_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_0_3_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_3_output, mlp_0_1_output):
        if attn_1_3_output in {")", "(", "<s>"}:
            return mlp_0_1_output == 2

    num_attn_2_0_pattern = select(mlp_0_1_outputs, attn_1_3_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_5_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_5_output, mlp_1_0_output):
        if attn_1_5_output in {"("}:
            return mlp_1_0_output == 19
        elif attn_1_5_output in {")"}:
            return mlp_1_0_output == 36
        elif attn_1_5_output in {"<s>"}:
            return mlp_1_0_output == 1

    num_attn_2_1_pattern = select(mlp_1_0_outputs, attn_1_5_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_6_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(mlp_1_0_output, mlp_0_0_output):
        if mlp_1_0_output in {0, 4}:
            return mlp_0_0_output == 37
        elif mlp_1_0_output in {1, 3, 6, 39, 19}:
            return mlp_0_0_output == 5
        elif mlp_1_0_output in {2, 5}:
            return mlp_0_0_output == 26
        elif mlp_1_0_output in {11, 28, 7}:
            return mlp_0_0_output == 14
        elif mlp_1_0_output in {8, 16, 32, 31}:
            return mlp_0_0_output == 2
        elif mlp_1_0_output in {9, 25}:
            return mlp_0_0_output == 24
        elif mlp_1_0_output in {10, 27}:
            return mlp_0_0_output == 16
        elif mlp_1_0_output in {12, 37}:
            return mlp_0_0_output == 4
        elif mlp_1_0_output in {13, 22, 30}:
            return mlp_0_0_output == 34
        elif mlp_1_0_output in {14}:
            return mlp_0_0_output == 25
        elif mlp_1_0_output in {23, 15}:
            return mlp_0_0_output == 35
        elif mlp_1_0_output in {17}:
            return mlp_0_0_output == 21
        elif mlp_1_0_output in {18, 36}:
            return mlp_0_0_output == 12
        elif mlp_1_0_output in {20}:
            return mlp_0_0_output == 0
        elif mlp_1_0_output in {21}:
            return mlp_0_0_output == 28
        elif mlp_1_0_output in {24}:
            return mlp_0_0_output == 23
        elif mlp_1_0_output in {26}:
            return mlp_0_0_output == 33
        elif mlp_1_0_output in {29}:
            return mlp_0_0_output == 6
        elif mlp_1_0_output in {33}:
            return mlp_0_0_output == 19
        elif mlp_1_0_output in {34}:
            return mlp_0_0_output == 15
        elif mlp_1_0_output in {35}:
            return mlp_0_0_output == 9
        elif mlp_1_0_output in {38}:
            return mlp_0_0_output == 36

    num_attn_2_2_pattern = select(mlp_0_0_outputs, mlp_1_0_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_6_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_1_output, mlp_0_1_output):
        if attn_1_1_output in {"("}:
            return mlp_0_1_output == 9
        elif attn_1_1_output in {")"}:
            return mlp_0_1_output == 7
        elif attn_1_1_output in {"<s>"}:
            return mlp_0_1_output == 2

    num_attn_2_3_pattern = select(mlp_0_1_outputs, attn_1_1_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_7_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(position, attn_0_1_output):
        if position in {
            0,
            4,
            6,
            8,
            10,
            12,
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
        }:
            return attn_0_1_output == ""
        elif position in {1, 2, 3, 5, 7, 9, 11, 13}:
            return attn_0_1_output == ")"

    num_attn_2_4_pattern = select(attn_0_1_outputs, positions, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_1_2_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(num_mlp_0_1_output, attn_1_6_output):
        if num_mlp_0_1_output in {
            0,
            1,
            2,
            3,
            5,
            8,
            9,
            10,
            11,
            13,
            14,
            15,
            16,
            19,
            21,
            22,
            23,
            25,
            26,
            29,
            31,
            32,
            34,
            35,
            36,
            37,
            38,
        }:
            return attn_1_6_output == ")"
        elif num_mlp_0_1_output in {33, 4, 6, 7, 18}:
            return attn_1_6_output == ""
        elif num_mlp_0_1_output in {39, 12, 17, 20, 24, 27, 28, 30}:
            return attn_1_6_output == "<s>"

    num_attn_2_5_pattern = select(
        attn_1_6_outputs, num_mlp_0_1_outputs, num_predicate_2_5
    )
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_7_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_0_1_output, mlp_0_0_output):
        if attn_0_1_output in {"("}:
            return mlp_0_0_output == 35
        elif attn_0_1_output in {")", "<s>"}:
            return mlp_0_0_output == 12

    num_attn_2_6_pattern = select(mlp_0_0_outputs, attn_0_1_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_2_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(mlp_0_0_output, mlp_1_0_output):
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
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            18,
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
            32,
            33,
            34,
            35,
            38,
        }:
            return mlp_1_0_output == 2
        elif mlp_0_0_output in {9, 19, 36, 17}:
            return mlp_1_0_output == 5
        elif mlp_0_0_output in {26}:
            return mlp_1_0_output == 39
        elif mlp_0_0_output in {37}:
            return mlp_1_0_output == 26
        elif mlp_0_0_output in {39}:
            return mlp_1_0_output == 28

    num_attn_2_7_pattern = select(mlp_1_0_outputs, mlp_0_0_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_5_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_3_output, attn_2_5_output):
        key = (attn_2_3_output, attn_2_5_output)
        if key in {(")", "<s>"), ("<s>", "<s>")}:
            return 23
        elif key in {(")", ")"), ("<s>", ")")}:
            return 15
        return 14

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_3_outputs, attn_2_5_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_7_output, attn_2_4_output):
        key = (attn_2_7_output, attn_2_4_output)
        if key in {("<s>", "("), ("<s>", "<s>")}:
            return 32
        return 18

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_7_outputs, attn_2_4_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_4_output, num_attn_0_1_output):
        key = (num_attn_1_4_output, num_attn_0_1_output)
        return 3

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_6_output, num_attn_1_0_output):
        key = (num_attn_2_6_output, num_attn_1_0_output)
        return 30

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_6_outputs, num_attn_1_0_outputs)
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
            "(",
            ")",
            ")",
            "(",
            "(",
            "(",
            ")",
            ")",
            ")",
            "(",
            ")",
            ")",
            ")",
            "(",
            ")",
            ")",
            "(",
            "(",
            "(",
        ]
    )
)
