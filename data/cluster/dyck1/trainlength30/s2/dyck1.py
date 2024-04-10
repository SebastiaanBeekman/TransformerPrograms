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
        "output/length/rasp/dyck1/trainlength30/s2/dyck1_weights.csv",
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
            return position == 10
        elif token in {"<s>"}:
            return position == 56

    attn_0_0_pattern = select_closest(positions, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 2

    attn_0_1_pattern = select_closest(positions, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 57, 40}:
            return k_position == 19
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {8, 16, 2}:
            return k_position == 6
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 38
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {58, 7}:
            return k_position == 23
        elif q_position in {9}:
            return k_position == 57
        elif q_position in {10}:
            return k_position == 5
        elif q_position in {11}:
            return k_position == 25
        elif q_position in {12}:
            return k_position == 27
        elif q_position in {33, 13, 41}:
            return k_position == 33
        elif q_position in {14, 31}:
            return k_position == 15
        elif q_position in {15}:
            return k_position == 36
        elif q_position in {17}:
            return k_position == 41
        elif q_position in {18, 53}:
            return k_position == 44
        elif q_position in {19}:
            return k_position == 31
        elif q_position in {42, 20, 22, 23, 24, 27, 29}:
            return k_position == 7
        elif q_position in {59, 21}:
            return k_position == 50
        elif q_position in {25}:
            return k_position == 39
        elif q_position in {32, 34, 38, 46, 49, 50, 26, 28}:
            return k_position == 9
        elif q_position in {30}:
            return k_position == 34
        elif q_position in {35}:
            return k_position == 52
        elif q_position in {36}:
            return k_position == 13
        elif q_position in {37}:
            return k_position == 48
        elif q_position in {47, 39}:
            return k_position == 53
        elif q_position in {43}:
            return k_position == 45
        elif q_position in {44}:
            return k_position == 55
        elif q_position in {45}:
            return k_position == 43
        elif q_position in {48}:
            return k_position == 54
        elif q_position in {51}:
            return k_position == 46
        elif q_position in {52, 55}:
            return k_position == 17
        elif q_position in {56, 54}:
            return k_position == 21

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 32, 10, 43, 46, 48, 53, 54, 57, 58}:
            return k_position == 9
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 35, 31}:
            return k_position == 36
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8, 33, 39}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16, 34}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {20, 45, 38}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {49, 52, 22}:
            return k_position == 21
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {25}:
            return k_position == 24
        elif q_position in {26, 30}:
            return k_position == 25
        elif q_position in {27, 37}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 27
        elif q_position in {29}:
            return k_position == 54
        elif q_position in {42, 36}:
            return k_position == 39
        elif q_position in {40, 41, 55}:
            return k_position == 37
        elif q_position in {44}:
            return k_position == 28
        elif q_position in {47}:
            return k_position == 50
        elif q_position in {50}:
            return k_position == 41
        elif q_position in {51}:
            return k_position == 45
        elif q_position in {56}:
            return k_position == 47
        elif q_position in {59}:
            return k_position == 29

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
            return position == 4
        elif token in {"<s>"}:
            return position == 2

    attn_0_4_pattern = select_closest(positions, tokens, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 42

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
            return position == 4
        elif token in {"<s>"}:
            return position == 30

    attn_0_6_pattern = select_closest(positions, tokens, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0, 18}:
            return k_position == 17
        elif q_position in {1}:
            return k_position == 37
        elif q_position in {2, 36, 37, 23}:
            return k_position == 20
        elif q_position in {34, 3}:
            return k_position == 41
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {40, 5}:
            return k_position == 13
        elif q_position in {6, 30}:
            return k_position == 5
        elif q_position in {53, 7}:
            return k_position == 33
        elif q_position in {8}:
            return k_position == 6
        elif q_position in {9}:
            return k_position == 7
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {26, 11}:
            return k_position == 8
        elif q_position in {41, 43, 12, 47, 48, 52}:
            return k_position == 11
        elif q_position in {13, 14}:
            return k_position == 10
        elif q_position in {15}:
            return k_position == 12
        elif q_position in {16, 38}:
            return k_position == 15
        elif q_position in {56, 17, 35, 45}:
            return k_position == 16
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {20, 21}:
            return k_position == 19
        elif q_position in {44, 22}:
            return k_position == 21
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {25, 50}:
            return k_position == 24
        elif q_position in {27}:
            return k_position == 4
        elif q_position in {28, 55}:
            return k_position == 25
        elif q_position in {29}:
            return k_position == 50
        elif q_position in {31}:
            return k_position == 56
        elif q_position in {32}:
            return k_position == 58
        elif q_position in {33}:
            return k_position == 32
        elif q_position in {39}:
            return k_position == 45
        elif q_position in {42}:
            return k_position == 36
        elif q_position in {46}:
            return k_position == 54
        elif q_position in {49}:
            return k_position == 39
        elif q_position in {51}:
            return k_position == 26
        elif q_position in {54}:
            return k_position == 29
        elif q_position in {57}:
            return k_position == 34
        elif q_position in {58}:
            return k_position == 38
        elif q_position in {59}:
            return k_position == 47

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"("}:
            return position == 9
        elif token in {")"}:
            return position == 45
        elif token in {"<s>"}:
            return position == 46

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(token, position):
        if token in {"("}:
            return position == 26
        elif token in {")"}:
            return position == 14
        elif token in {"<s>"}:
            return position == 3

    num_attn_0_1_pattern = select(positions, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
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
            33,
            35,
            36,
            39,
            46,
            49,
            51,
            54,
        }:
            return token == ""
        elif position in {
            1,
            3,
            4,
            6,
            32,
            34,
            37,
            38,
            40,
            41,
            42,
            43,
            44,
            45,
            47,
            48,
            50,
            52,
            53,
            55,
            56,
            57,
            58,
            59,
        }:
            return token == ")"
        elif position in {16}:
            return token == "<pad>"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
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
            8,
            10,
            12,
            14,
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
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
        }:
            return token == ""
        elif position in {1, 3, 5, 7, 9}:
            return token == ")"
        elif position in {27, 11, 13, 15}:
            return token == "<s>"
        elif position in {20}:
            return token == "<pad>"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(token, position):
        if token in {"("}:
            return position == 56
        elif token in {")"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 7

    num_attn_0_4_pattern = select(positions, tokens, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {
            0,
            6,
            7,
            8,
            9,
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
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            58,
            59,
        }:
            return token == ""
        elif position in {1, 2, 3, 4, 5, 34, 37, 42}:
            return token == ")"
        elif position in {10, 14}:
            return token == "<s>"
        elif position in {57, 38}:
            return token == "<pad>"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(q_position, k_position):
        if q_position in {0, 52}:
            return k_position == 56
        elif q_position in {1, 20, 21}:
            return k_position == 34
        elif q_position in {2, 53}:
            return k_position == 13
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4, 12}:
            return k_position == 23
        elif q_position in {5}:
            return k_position == 16
        elif q_position in {6}:
            return k_position == 52
        elif q_position in {41, 7}:
            return k_position == 8
        elif q_position in {8, 51, 44}:
            return k_position == 50
        elif q_position in {9}:
            return k_position == 12
        elif q_position in {10}:
            return k_position == 33
        elif q_position in {27, 11}:
            return k_position == 32
        elif q_position in {13, 39}:
            return k_position == 27
        elif q_position in {14}:
            return k_position == 6
        elif q_position in {15}:
            return k_position == 17
        elif q_position in {16}:
            return k_position == 45
        elif q_position in {17, 18}:
            return k_position == 25
        elif q_position in {19}:
            return k_position == 19
        elif q_position in {57, 22, 47}:
            return k_position == 35
        elif q_position in {59, 28, 23}:
            return k_position == 53
        elif q_position in {24}:
            return k_position == 48
        elif q_position in {54, 25, 30}:
            return k_position == 44
        elif q_position in {26, 38}:
            return k_position == 42
        elif q_position in {29}:
            return k_position == 37
        elif q_position in {33, 55, 31}:
            return k_position == 1
        elif q_position in {32}:
            return k_position == 26
        elif q_position in {34}:
            return k_position == 36
        elif q_position in {56, 35}:
            return k_position == 58
        elif q_position in {36}:
            return k_position == 41
        elif q_position in {37}:
            return k_position == 47
        elif q_position in {40, 49, 50}:
            return k_position == 14
        elif q_position in {42}:
            return k_position == 39
        elif q_position in {43}:
            return k_position == 11
        elif q_position in {45, 46}:
            return k_position == 38
        elif q_position in {48}:
            return k_position == 28
        elif q_position in {58}:
            return k_position == 30

    num_attn_0_6_pattern = select(positions, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(q_token, k_token):
        if q_token in {"(", "<s>", ")"}:
            return k_token == ""

    num_attn_0_7_pattern = select(tokens, tokens, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_7_output, attn_0_3_output):
        key = (attn_0_7_output, attn_0_3_output)
        if key in {("(", "("), ("(", "<s>")}:
            return 39
        elif key in {
            ("(", ")"),
            (")", ")"),
            (")", "<s>"),
            ("<s>", ")"),
            ("<s>", "<s>"),
        }:
            return 2
        return 27

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_7_outputs, attn_0_3_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_6_output, attn_0_1_output):
        key = (attn_0_6_output, attn_0_1_output)
        if key in {(")", ")"), (")", "<s>"), ("<s>", ")"), ("<s>", "<s>")}:
            return 2
        return 13

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_1_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output):
        key = num_attn_0_2_output
        return 8

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_4_output, num_attn_0_7_output):
        key = (num_attn_0_4_output, num_attn_0_7_output)
        return 6

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_4_outputs, num_attn_0_7_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_1_output, position):
        if attn_0_1_output in {"(", "<s>"}:
            return position == 1
        elif attn_0_1_output in {")"}:
            return position == 3

    attn_1_0_pattern = select_closest(positions, attn_0_1_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_3_output, position):
        if attn_0_3_output in {"(", "<s>"}:
            return position == 1
        elif attn_0_3_output in {")"}:
            return position == 3

    attn_1_1_pattern = select_closest(positions, attn_0_3_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_2_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(token, position):
        if token in {"("}:
            return position == 7
        elif token in {")"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 5

    attn_1_2_pattern = select_closest(positions, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_6_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 53

    attn_1_3_pattern = select_closest(positions, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 7
        elif token in {"<s>"}:
            return position == 45

    attn_1_4_pattern = select_closest(positions, tokens, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, tokens)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 12
        elif token in {"<s>"}:
            return position == 3

    attn_1_5_pattern = select_closest(positions, tokens, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_5_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(token, mlp_0_1_output):
        if token in {"("}:
            return mlp_0_1_output == 1
        elif token in {")"}:
            return mlp_0_1_output == 13
        elif token in {"<s>"}:
            return mlp_0_1_output == 3

    attn_1_6_pattern = select_closest(mlp_0_1_outputs, tokens, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_1_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 10
        elif token in {"<s>"}:
            return position == 5

    attn_1_7_pattern = select_closest(positions, tokens, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_1_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(num_mlp_0_0_output, token):
        if num_mlp_0_0_output in {0}:
            return token == "<s>"
        elif num_mlp_0_0_output in {
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
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
        }:
            return token == ""
        elif num_mlp_0_0_output in {9}:
            return token == "<pad>"
        elif num_mlp_0_0_output in {40, 41}:
            return token == "("

    num_attn_1_0_pattern = select(tokens, num_mlp_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_4_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(num_mlp_0_0_output, num_mlp_0_1_output):
        if num_mlp_0_0_output in {0, 7}:
            return num_mlp_0_1_output == 10
        elif num_mlp_0_0_output in {1, 4, 36, 59, 22, 27}:
            return num_mlp_0_1_output == 29
        elif num_mlp_0_0_output in {2, 5}:
            return num_mlp_0_1_output == 53
        elif num_mlp_0_0_output in {32, 3, 8, 43, 21}:
            return num_mlp_0_1_output == 49
        elif num_mlp_0_0_output in {58, 6, 48, 53, 26, 31}:
            return num_mlp_0_1_output == 8
        elif num_mlp_0_0_output in {9}:
            return num_mlp_0_1_output == 18
        elif num_mlp_0_0_output in {10}:
            return num_mlp_0_1_output == 40
        elif num_mlp_0_0_output in {11}:
            return num_mlp_0_1_output == 55
        elif num_mlp_0_0_output in {12}:
            return num_mlp_0_1_output == 36
        elif num_mlp_0_0_output in {34, 40, 13, 46, 15, 23, 29}:
            return num_mlp_0_1_output == 6
        elif num_mlp_0_0_output in {14}:
            return num_mlp_0_1_output == 24
        elif num_mlp_0_0_output in {33, 37, 16, 20, 56}:
            return num_mlp_0_1_output == 42
        elif num_mlp_0_0_output in {17}:
            return num_mlp_0_1_output == 56
        elif num_mlp_0_0_output in {38, 42, 45, 18, 19, 52}:
            return num_mlp_0_1_output == 44
        elif num_mlp_0_0_output in {24}:
            return num_mlp_0_1_output == 4
        elif num_mlp_0_0_output in {25}:
            return num_mlp_0_1_output == 21
        elif num_mlp_0_0_output in {50, 28}:
            return num_mlp_0_1_output == 35
        elif num_mlp_0_0_output in {30}:
            return num_mlp_0_1_output == 16
        elif num_mlp_0_0_output in {35}:
            return num_mlp_0_1_output == 50
        elif num_mlp_0_0_output in {39}:
            return num_mlp_0_1_output == 28
        elif num_mlp_0_0_output in {41}:
            return num_mlp_0_1_output == 52
        elif num_mlp_0_0_output in {49, 44}:
            return num_mlp_0_1_output == 22
        elif num_mlp_0_0_output in {47}:
            return num_mlp_0_1_output == 51
        elif num_mlp_0_0_output in {57, 51, 55}:
            return num_mlp_0_1_output == 58
        elif num_mlp_0_0_output in {54}:
            return num_mlp_0_1_output == 0

    num_attn_1_1_pattern = select(
        num_mlp_0_1_outputs, num_mlp_0_0_outputs, num_predicate_1_1
    )
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_2_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_0_output, mlp_0_1_output):
        if attn_0_0_output in {"(", "<s>", ")"}:
            return mlp_0_1_output == 2

    num_attn_1_2_pattern = select(mlp_0_1_outputs, attn_0_0_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_3_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(num_mlp_0_1_output, mlp_0_1_output):
        if num_mlp_0_1_output in {
            0,
            1,
            3,
            4,
            5,
            6,
            8,
            12,
            13,
            14,
            18,
            20,
            22,
            24,
            26,
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
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
        }:
            return mlp_0_1_output == 2
        elif num_mlp_0_1_output in {2, 10, 11, 16, 21, 23}:
            return mlp_0_1_output == 20
        elif num_mlp_0_1_output in {43, 7}:
            return mlp_0_1_output == 13
        elif num_mlp_0_1_output in {9}:
            return mlp_0_1_output == 55
        elif num_mlp_0_1_output in {15}:
            return mlp_0_1_output == 8
        elif num_mlp_0_1_output in {17, 27}:
            return mlp_0_1_output == 27
        elif num_mlp_0_1_output in {19}:
            return mlp_0_1_output == 21
        elif num_mlp_0_1_output in {25}:
            return mlp_0_1_output == 46
        elif num_mlp_0_1_output in {39}:
            return mlp_0_1_output == 39
        elif num_mlp_0_1_output in {45}:
            return mlp_0_1_output == 54

    num_attn_1_3_pattern = select(
        mlp_0_1_outputs, num_mlp_0_1_outputs, num_predicate_1_3
    )
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_5_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(attn_0_0_output, mlp_0_0_output):
        if attn_0_0_output in {"(", "<s>", ")"}:
            return mlp_0_0_output == 2

    num_attn_1_4_pattern = select(mlp_0_0_outputs, attn_0_0_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_4_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(num_mlp_0_1_output, mlp_0_0_output):
        if num_mlp_0_1_output in {
            0,
            1,
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
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
        }:
            return mlp_0_0_output == 39
        elif num_mlp_0_1_output in {2}:
            return mlp_0_0_output == 57
        elif num_mlp_0_1_output in {27, 5}:
            return mlp_0_0_output == 7
        elif num_mlp_0_1_output in {14}:
            return mlp_0_0_output == 44
        elif num_mlp_0_1_output in {19}:
            return mlp_0_0_output == 24
        elif num_mlp_0_1_output in {28}:
            return mlp_0_0_output == 20
        elif num_mlp_0_1_output in {40}:
            return mlp_0_0_output == 4

    num_attn_1_5_pattern = select(
        mlp_0_0_outputs, num_mlp_0_1_outputs, num_predicate_1_5
    )
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_0_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(num_mlp_0_1_output, mlp_0_0_output):
        if num_mlp_0_1_output in {0, 41}:
            return mlp_0_0_output == 58
        elif num_mlp_0_1_output in {1, 58, 23}:
            return mlp_0_0_output == 52
        elif num_mlp_0_1_output in {32, 2, 3, 36, 9, 53, 54, 55, 56, 57, 59, 30, 31}:
            return mlp_0_0_output == 27
        elif num_mlp_0_1_output in {
            4,
            5,
            6,
            7,
            8,
            10,
            42,
            12,
            43,
            14,
            44,
            49,
            50,
            51,
            52,
        }:
            return mlp_0_0_output == 2
        elif num_mlp_0_1_output in {11}:
            return mlp_0_0_output == 20
        elif num_mlp_0_1_output in {19, 13}:
            return mlp_0_0_output == 40
        elif num_mlp_0_1_output in {26, 15}:
            return mlp_0_0_output == 24
        elif num_mlp_0_1_output in {16, 48}:
            return mlp_0_0_output == 12
        elif num_mlp_0_1_output in {17}:
            return mlp_0_0_output == 28
        elif num_mlp_0_1_output in {18}:
            return mlp_0_0_output == 57
        elif num_mlp_0_1_output in {20}:
            return mlp_0_0_output == 44
        elif num_mlp_0_1_output in {21}:
            return mlp_0_0_output == 45
        elif num_mlp_0_1_output in {22}:
            return mlp_0_0_output == 30
        elif num_mlp_0_1_output in {24, 46}:
            return mlp_0_0_output == 49
        elif num_mlp_0_1_output in {25, 27}:
            return mlp_0_0_output == 36
        elif num_mlp_0_1_output in {28}:
            return mlp_0_0_output == 19
        elif num_mlp_0_1_output in {37, 29}:
            return mlp_0_0_output == 31
        elif num_mlp_0_1_output in {33}:
            return mlp_0_0_output == 38
        elif num_mlp_0_1_output in {34}:
            return mlp_0_0_output == 32
        elif num_mlp_0_1_output in {35, 38}:
            return mlp_0_0_output == 43
        elif num_mlp_0_1_output in {39}:
            return mlp_0_0_output == 22
        elif num_mlp_0_1_output in {40}:
            return mlp_0_0_output == 54
        elif num_mlp_0_1_output in {45}:
            return mlp_0_0_output == 59
        elif num_mlp_0_1_output in {47}:
            return mlp_0_0_output == 16

    num_attn_1_6_pattern = select(
        mlp_0_0_outputs, num_mlp_0_1_outputs, num_predicate_1_6
    )
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_2_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(attn_0_6_output, mlp_0_0_output):
        if attn_0_6_output in {"(", "<s>", ")"}:
            return mlp_0_0_output == 27

    num_attn_1_7_pattern = select(mlp_0_0_outputs, attn_0_6_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_4_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_0_output, attn_0_3_output):
        key = (attn_0_0_output, attn_0_3_output)
        if key in {(")", ")"), (")", "<s>"), ("<s>", ")"), ("<s>", "<s>")}:
            return 14
        elif key in {("(", "("), ("<s>", "(")}:
            return 9
        return 23

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_3_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_1_output, attn_1_7_output):
        key = (attn_1_1_output, attn_1_7_output)
        return 5

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_1_7_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_6_output, num_attn_1_5_output):
        key = (num_attn_1_6_output, num_attn_1_5_output)
        return 18

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_1_output, num_attn_1_7_output):
        key = (num_attn_1_1_output, num_attn_1_7_output)
        return 21

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_7_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(token, position):
        if token in {"("}:
            return position == 7
        elif token in {")"}:
            return position == 3
        elif token in {"<s>"}:
            return position == 5

    attn_2_0_pattern = select_closest(positions, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_4_output, position):
        if attn_0_4_output in {"("}:
            return position == 3
        elif attn_0_4_output in {")"}:
            return position == 11
        elif attn_0_4_output in {"<s>"}:
            return position == 2

    attn_2_1_pattern = select_closest(positions, attn_0_4_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_position, k_position):
        if q_position in {0, 55, 14, 22}:
            return k_position == 13
        elif q_position in {1, 2, 35, 37, 44, 45, 46, 47, 49, 58, 31}:
            return k_position == 1
        elif q_position in {3, 39}:
            return k_position == 2
        elif q_position in {4, 5}:
            return k_position == 3
        elif q_position in {33, 34, 6, 48, 51, 54, 56, 57, 30}:
            return k_position == 5
        elif q_position in {29, 12, 23, 7}:
            return k_position == 6
        elif q_position in {32, 8, 9, 10, 11, 13, 15, 19, 25}:
            return k_position == 7
        elif q_position in {16, 17, 18, 20, 21, 27}:
            return k_position == 11
        elif q_position in {24, 53}:
            return k_position == 12
        elif q_position in {26}:
            return k_position == 14
        elif q_position in {41, 28}:
            return k_position == 20
        elif q_position in {59, 36}:
            return k_position == 16
        elif q_position in {38}:
            return k_position == 46
        elif q_position in {40}:
            return k_position == 8
        elif q_position in {42}:
            return k_position == 15
        elif q_position in {43}:
            return k_position == 47
        elif q_position in {50}:
            return k_position == 4
        elif q_position in {52}:
            return k_position == 44

    attn_2_2_pattern = select_closest(positions, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_4_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_5_output, position):
        if attn_0_5_output in {"(", "<s>", ")"}:
            return position == 2

    attn_2_3_pattern = select_closest(positions, attn_0_5_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_3_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_4_output, position):
        if attn_0_4_output in {"(", ")"}:
            return position == 7
        elif attn_0_4_output in {"<s>"}:
            return position == 1

    attn_2_4_pattern = select_closest(positions, attn_0_4_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_7_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_4_output, position):
        if attn_0_4_output in {"("}:
            return position == 3
        elif attn_0_4_output in {")"}:
            return position == 13
        elif attn_0_4_output in {"<s>"}:
            return position == 5

    attn_2_5_pattern = select_closest(positions, attn_0_4_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_1_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_0_1_output, position):
        if attn_0_1_output in {"("}:
            return position == 3
        elif attn_0_1_output in {"<s>", ")"}:
            return position == 5

    attn_2_6_pattern = select_closest(positions, attn_0_1_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_5_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_0_2_output, position):
        if attn_0_2_output in {"("}:
            return position == 3
        elif attn_0_2_output in {")"}:
            return position == 11
        elif attn_0_2_output in {"<s>"}:
            return position == 6

    attn_2_7_pattern = select_closest(positions, attn_0_2_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_4_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_2_output, attn_1_1_output):
        if attn_1_2_output in {"(", "<s>", ")"}:
            return attn_1_1_output == ""

    num_attn_2_0_pattern = select(attn_1_1_outputs, attn_1_2_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_4_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_0_output, mlp_1_0_output):
        if attn_1_0_output in {"(", "<s>", ")"}:
            return mlp_1_0_output == 14

    num_attn_2_1_pattern = select(mlp_1_0_outputs, attn_1_0_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_3_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_7_output, token):
        if attn_1_7_output in {"(", "<s>", ")"}:
            return token == ""

    num_attn_2_2_pattern = select(tokens, attn_1_7_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_5_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(mlp_1_1_output, mlp_0_1_output):
        if mlp_1_1_output in {0}:
            return mlp_0_1_output == 41
        elif mlp_1_1_output in {1}:
            return mlp_0_1_output == 11
        elif mlp_1_1_output in {2, 23}:
            return mlp_0_1_output == 9
        elif mlp_1_1_output in {3}:
            return mlp_0_1_output == 8
        elif mlp_1_1_output in {32, 4, 11, 13, 46, 15, 19, 24, 25}:
            return mlp_0_1_output == 22
        elif mlp_1_1_output in {16, 17, 5}:
            return mlp_0_1_output == 53
        elif mlp_1_1_output in {6, 22}:
            return mlp_0_1_output == 45
        elif mlp_1_1_output in {20, 7}:
            return mlp_0_1_output == 32
        elif mlp_1_1_output in {8, 43}:
            return mlp_0_1_output == 31
        elif mlp_1_1_output in {56, 9, 14}:
            return mlp_0_1_output == 39
        elif mlp_1_1_output in {26, 10, 18, 38}:
            return mlp_0_1_output == 58
        elif mlp_1_1_output in {12, 53}:
            return mlp_0_1_output == 35
        elif mlp_1_1_output in {33, 36, 21}:
            return mlp_0_1_output == 23
        elif mlp_1_1_output in {27}:
            return mlp_0_1_output == 15
        elif mlp_1_1_output in {41, 28}:
            return mlp_0_1_output == 13
        elif mlp_1_1_output in {52, 29}:
            return mlp_0_1_output == 47
        elif mlp_1_1_output in {37, 30}:
            return mlp_0_1_output == 6
        elif mlp_1_1_output in {42, 31}:
            return mlp_0_1_output == 33
        elif mlp_1_1_output in {34}:
            return mlp_0_1_output == 54
        elif mlp_1_1_output in {35}:
            return mlp_0_1_output == 30
        elif mlp_1_1_output in {58, 39}:
            return mlp_0_1_output == 36
        elif mlp_1_1_output in {40, 59}:
            return mlp_0_1_output == 28
        elif mlp_1_1_output in {44, 55}:
            return mlp_0_1_output == 29
        elif mlp_1_1_output in {48, 45}:
            return mlp_0_1_output == 1
        elif mlp_1_1_output in {47}:
            return mlp_0_1_output == 44
        elif mlp_1_1_output in {49}:
            return mlp_0_1_output == 42
        elif mlp_1_1_output in {50}:
            return mlp_0_1_output == 50
        elif mlp_1_1_output in {51}:
            return mlp_0_1_output == 59
        elif mlp_1_1_output in {54}:
            return mlp_0_1_output == 48
        elif mlp_1_1_output in {57}:
            return mlp_0_1_output == 56

    num_attn_2_3_pattern = select(mlp_0_1_outputs, mlp_1_1_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_5_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(mlp_1_0_output, num_mlp_0_1_output):
        if mlp_1_0_output in {0, 43}:
            return num_mlp_0_1_output == 13
        elif mlp_1_0_output in {1}:
            return num_mlp_0_1_output == 17
        elif mlp_1_0_output in {2}:
            return num_mlp_0_1_output == 26
        elif mlp_1_0_output in {25, 3}:
            return num_mlp_0_1_output == 15
        elif mlp_1_0_output in {4, 31}:
            return num_mlp_0_1_output == 28
        elif mlp_1_0_output in {12, 5}:
            return num_mlp_0_1_output == 53
        elif mlp_1_0_output in {16, 6}:
            return num_mlp_0_1_output == 25
        elif mlp_1_0_output in {7}:
            return num_mlp_0_1_output == 27
        elif mlp_1_0_output in {8}:
            return num_mlp_0_1_output == 24
        elif mlp_1_0_output in {9, 22}:
            return num_mlp_0_1_output == 32
        elif mlp_1_0_output in {10}:
            return num_mlp_0_1_output == 47
        elif mlp_1_0_output in {19, 11}:
            return num_mlp_0_1_output == 44
        elif mlp_1_0_output in {13}:
            return num_mlp_0_1_output == 54
        elif mlp_1_0_output in {52, 14}:
            return num_mlp_0_1_output == 55
        elif mlp_1_0_output in {15}:
            return num_mlp_0_1_output == 37
        elif mlp_1_0_output in {48, 17, 51, 21}:
            return num_mlp_0_1_output == 16
        elif mlp_1_0_output in {18}:
            return num_mlp_0_1_output == 5
        elif mlp_1_0_output in {59, 20, 38}:
            return num_mlp_0_1_output == 29
        elif mlp_1_0_output in {30, 23}:
            return num_mlp_0_1_output == 22
        elif mlp_1_0_output in {24, 50}:
            return num_mlp_0_1_output == 56
        elif mlp_1_0_output in {26, 58}:
            return num_mlp_0_1_output == 20
        elif mlp_1_0_output in {33, 27}:
            return num_mlp_0_1_output == 52
        elif mlp_1_0_output in {35, 28, 53}:
            return num_mlp_0_1_output == 19
        elif mlp_1_0_output in {29}:
            return num_mlp_0_1_output == 33
        elif mlp_1_0_output in {32}:
            return num_mlp_0_1_output == 31
        elif mlp_1_0_output in {34, 36}:
            return num_mlp_0_1_output == 48
        elif mlp_1_0_output in {57, 37}:
            return num_mlp_0_1_output == 11
        elif mlp_1_0_output in {39}:
            return num_mlp_0_1_output == 58
        elif mlp_1_0_output in {40}:
            return num_mlp_0_1_output == 41
        elif mlp_1_0_output in {41}:
            return num_mlp_0_1_output == 34
        elif mlp_1_0_output in {42}:
            return num_mlp_0_1_output == 35
        elif mlp_1_0_output in {44}:
            return num_mlp_0_1_output == 38
        elif mlp_1_0_output in {45}:
            return num_mlp_0_1_output == 45
        elif mlp_1_0_output in {46}:
            return num_mlp_0_1_output == 8
        elif mlp_1_0_output in {47}:
            return num_mlp_0_1_output == 23
        elif mlp_1_0_output in {49}:
            return num_mlp_0_1_output == 30
        elif mlp_1_0_output in {54}:
            return num_mlp_0_1_output == 51
        elif mlp_1_0_output in {55}:
            return num_mlp_0_1_output == 42
        elif mlp_1_0_output in {56}:
            return num_mlp_0_1_output == 49

    num_attn_2_4_pattern = select(
        num_mlp_0_1_outputs, mlp_1_0_outputs, num_predicate_2_4
    )
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_1_1_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(mlp_1_1_output, attn_1_1_output):
        if mlp_1_1_output in {
            0,
            5,
            7,
            19,
            20,
            21,
            30,
            34,
            35,
            37,
            38,
            39,
            40,
            42,
            43,
            44,
            49,
            50,
            55,
            58,
            59,
        }:
            return attn_1_1_output == ")"
        elif mlp_1_1_output in {
            1,
            2,
            4,
            6,
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
            22,
            23,
            24,
            25,
            27,
            28,
            29,
            31,
            32,
            33,
            36,
            41,
            45,
            46,
            47,
            48,
            51,
            52,
            53,
            54,
            56,
            57,
        }:
            return attn_1_1_output == ""
        elif mlp_1_1_output in {26, 3}:
            return attn_1_1_output == "<s>"

    num_attn_2_5_pattern = select(attn_1_1_outputs, mlp_1_1_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_5_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_7_output, mlp_1_0_output):
        if attn_1_7_output in {"("}:
            return mlp_1_0_output == 38
        elif attn_1_7_output in {")"}:
            return mlp_1_0_output == 5
        elif attn_1_7_output in {"<s>"}:
            return mlp_1_0_output == 14

    num_attn_2_6_pattern = select(mlp_1_0_outputs, attn_1_7_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_3_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_1_7_output, token):
        if attn_1_7_output in {"(", "<s>", ")"}:
            return token == ""

    num_attn_2_7_pattern = select(tokens, attn_1_7_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_1_6_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_1_output, attn_2_3_output):
        key = (attn_2_1_output, attn_2_3_output)
        if key in {
            (")", "("),
            (")", ")"),
            (")", "<s>"),
            ("<s>", "("),
            ("<s>", ")"),
            ("<s>", "<s>"),
        }:
            return 2
        elif key in {("(", ")")}:
            return 15
        return 30

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_1_outputs, attn_2_3_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_4_output, attn_2_1_output):
        key = (attn_2_4_output, attn_2_1_output)
        if key in {("(", ")"), (")", ")"), (")", "<s>"), ("<s>", ")")}:
            return 47
        return 37

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_4_outputs, attn_2_1_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_1_output, num_attn_2_4_output):
        key = (num_attn_2_1_output, num_attn_2_4_output)
        return 14

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_2_4_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_2_output, num_attn_1_7_output):
        key = (num_attn_1_2_output, num_attn_1_7_output)
        return 1

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_7_outputs)
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
            "(",
            ")",
            "(",
            ")",
            "(",
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
            "(",
            "(",
            ")",
            ")",
            ")",
            "(",
            "(",
            "(",
            ")",
            ")",
        ]
    )
)
