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
        "output/length/rasp/dyck2/trainlength40/s3/dyck2_weights.csv",
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
        if position in {
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            9,
            10,
            36,
            42,
            43,
            44,
            45,
            46,
            49,
            54,
            55,
            56,
            59,
            60,
            61,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            72,
            73,
            75,
            79,
        }:
            return token == "}"
        elif position in {34, 8, 41, 11, 12, 57}:
            return token == ")"
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
            37,
            38,
            40,
            47,
            48,
            50,
            51,
            52,
            53,
            58,
            62,
            70,
            71,
            74,
            76,
            77,
            78,
        }:
            return token == ""
        elif position in {35, 39}:
            return token == "{"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_token, k_token):
        if q_token in {"(", ")", "{"}:
            return k_token == ")"
        elif q_token in {"<s>"}:
            return k_token == ""
        elif q_token in {"}"}:
            return k_token == "}"

    attn_0_1_pattern = select_closest(tokens, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {
            0,
            7,
            9,
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
            59,
            61,
            62,
            63,
            65,
            67,
            68,
            69,
            71,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
        }:
            return k_position == 6
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {8, 70, 66, 6}:
            return k_position == 5
        elif q_position in {64, 72, 10, 42, 57, 58, 60}:
            return k_position == 7
        elif q_position in {11}:
            return k_position == 8
        elif q_position in {34, 12}:
            return k_position == 11
        elif q_position in {36, 13, 38}:
            return k_position == 12
        elif q_position in {14, 39}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {25}:
            return k_position == 24
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 27
        elif q_position in {29}:
            return k_position == 28
        elif q_position in {30}:
            return k_position == 29
        elif q_position in {31}:
            return k_position == 30
        elif q_position in {32}:
            return k_position == 10
        elif q_position in {33}:
            return k_position == 32
        elif q_position in {35}:
            return k_position == 34
        elif q_position in {37}:
            return k_position == 36

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {
            0,
            9,
            40,
            41,
            43,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            53,
            54,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            65,
            66,
            68,
            71,
            74,
            76,
            77,
            78,
            79,
        }:
            return k_position == 6
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 39
        elif q_position in {33, 3}:
            return k_position == 4
        elif q_position in {4, 5}:
            return k_position == 0
        elif q_position in {6, 7}:
            return k_position == 5
        elif q_position in {
            64,
            67,
            69,
            70,
            8,
            72,
            10,
            11,
            42,
            44,
            14,
            73,
            75,
            52,
            55,
            63,
        }:
            return k_position == 7
        elif q_position in {12, 13}:
            return k_position == 8
        elif q_position in {16, 15}:
            return k_position == 9
        elif q_position in {17}:
            return k_position == 14
        elif q_position in {18, 19, 22}:
            return k_position == 17
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {29, 21}:
            return k_position == 20
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {25}:
            return k_position == 24
        elif q_position in {32, 34, 36, 38, 26, 30}:
            return k_position == 11
        elif q_position in {27, 39}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 10
        elif q_position in {31}:
            return k_position == 30
        elif q_position in {35}:
            return k_position == 12
        elif q_position in {37}:
            return k_position == 36

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0, 51, 31}:
            return k_position == 10
        elif q_position in {1, 3}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 19
        elif q_position in {4}:
            return k_position == 4
        elif q_position in {9, 5, 46, 71}:
            return k_position == 40
        elif q_position in {18, 26, 6, 7}:
            return k_position == 3
        elif q_position in {8}:
            return k_position == 42
        elif q_position in {10}:
            return k_position == 73
        elif q_position in {16, 11}:
            return k_position == 15
        elif q_position in {12}:
            return k_position == 8
        elif q_position in {13}:
            return k_position == 43
        elif q_position in {14}:
            return k_position == 7
        elif q_position in {40, 27, 22, 15}:
            return k_position == 11
        elif q_position in {24, 17, 30}:
            return k_position == 0
        elif q_position in {19}:
            return k_position == 5
        elif q_position in {20}:
            return k_position == 18
        elif q_position in {21}:
            return k_position == 13
        elif q_position in {23}:
            return k_position == 16
        elif q_position in {25}:
            return k_position == 1
        elif q_position in {28}:
            return k_position == 14
        elif q_position in {48, 56, 29}:
            return k_position == 6
        elif q_position in {32}:
            return k_position == 69
        elif q_position in {33}:
            return k_position == 9
        elif q_position in {34, 59, 42, 70}:
            return k_position == 39
        elif q_position in {35}:
            return k_position == 26
        elif q_position in {58, 36}:
            return k_position == 70
        elif q_position in {72, 37}:
            return k_position == 64
        elif q_position in {69, 38, 49, 52, 53, 55}:
            return k_position == 61
        elif q_position in {62, 39}:
            return k_position == 58
        elif q_position in {41}:
            return k_position == 63
        elif q_position in {50, 43, 54}:
            return k_position == 71
        elif q_position in {44}:
            return k_position == 12
        elif q_position in {45}:
            return k_position == 62
        elif q_position in {47}:
            return k_position == 51
        elif q_position in {57}:
            return k_position == 59
        elif q_position in {60, 61}:
            return k_position == 49
        elif q_position in {63}:
            return k_position == 48
        elif q_position in {64, 67}:
            return k_position == 75
        elif q_position in {65}:
            return k_position == 74
        elif q_position in {66}:
            return k_position == 47
        elif q_position in {68}:
            return k_position == 60
        elif q_position in {73}:
            return k_position == 65
        elif q_position in {74}:
            return k_position == 79
        elif q_position in {75}:
            return k_position == 46
        elif q_position in {76, 79}:
            return k_position == 55
        elif q_position in {77}:
            return k_position == 78
        elif q_position in {78}:
            return k_position == 72

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(token, position):
        if token in {"("}:
            return position == 35
        elif token in {")"}:
            return position == 43
        elif token in {"<s>"}:
            return position == 27
        elif token in {"{"}:
            return position == 38
        elif token in {"}"}:
            return position == 77

    num_attn_0_1_pattern = select(positions, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {
            0,
            1,
            2,
            3,
            5,
            9,
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
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
        }:
            return token == ""
        elif position in {4, 68}:
            return token == "<pad>"
        elif position in {8, 13, 6, 7}:
            return token == "<s>"
        elif position in {11}:
            return token == "("

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"("}:
            return position == 37
        elif token in {")"}:
            return position == 12
        elif token in {"<s>"}:
            return position == 54
        elif token in {"{"}:
            return position == 38
        elif token in {"}"}:
            return position == 14

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_2_output, attn_0_1_output):
        key = (attn_0_2_output, attn_0_1_output)
        if key in {
            ("(", ")"),
            (")", "("),
            (")", ")"),
            (")", "<s>"),
            (")", "{"),
            ("<s>", ")"),
            ("{", ")"),
            ("}", "("),
            ("}", ")"),
            ("}", "<s>"),
            ("}", "{"),
        }:
            return 33
        elif key in {("<s>", "}"), ("}", "}")}:
            return 40
        elif key in {("{", "}")}:
            return 17
        elif key in {("(", "}")}:
            return 68
        elif key in {(")", "}")}:
            return 2
        return 4

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_1_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_3_output, attn_0_1_output):
        key = (attn_0_3_output, attn_0_1_output)
        if key in {(")", "("), ("<s>", "{")}:
            return 67
        elif key in {("(", "(")}:
            return 65
        elif key in {("<s>", "("), ("<s>", "<s>")}:
            return 25
        return 5

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_1_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_1_output, attn_0_0_output):
        key = (attn_0_1_output, attn_0_0_output)
        if key in {
            ("(", ")"),
            ("(", "}"),
            (")", "("),
            (")", ")"),
            (")", "<s>"),
            (")", "{"),
            (")", "}"),
            ("<s>", ")"),
            ("{", ")"),
            ("{", "}"),
            ("}", "("),
            ("}", ")"),
            ("}", "<s>"),
            ("}", "{"),
            ("}", "}"),
        }:
            return 5
        elif key in {("<s>", "}")}:
            return 37
        return 24

    mlp_0_2_outputs = [
        mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_0_outputs)
    ]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_1_output, attn_0_0_output):
        key = (attn_0_1_output, attn_0_0_output)
        return 2

    mlp_0_3_outputs = [
        mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_0_outputs)
    ]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output):
        key = num_attn_0_1_output
        return 21

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_1_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_2_output):
        key = num_attn_0_2_output
        return 21

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_0_output, num_attn_0_3_output):
        key = (num_attn_0_0_output, num_attn_0_3_output)
        if key in {
            (0, 25),
            (0, 26),
            (0, 27),
            (0, 28),
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
            (0, 40),
            (0, 41),
            (0, 42),
            (0, 43),
            (0, 44),
            (0, 45),
            (0, 46),
            (0, 47),
            (0, 48),
            (0, 49),
            (0, 50),
            (0, 51),
            (0, 52),
            (0, 53),
            (0, 54),
            (0, 55),
            (0, 56),
            (0, 57),
            (0, 58),
            (0, 59),
            (0, 60),
            (0, 61),
            (0, 62),
            (0, 63),
            (0, 64),
            (0, 65),
            (0, 66),
            (0, 67),
            (0, 68),
            (0, 69),
            (0, 70),
            (0, 71),
            (0, 72),
            (0, 73),
            (0, 74),
            (0, 75),
            (0, 76),
            (0, 77),
            (0, 78),
            (0, 79),
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
            (1, 50),
            (1, 51),
            (1, 52),
            (1, 53),
            (1, 54),
            (1, 55),
            (1, 56),
            (1, 57),
            (1, 58),
            (1, 59),
            (1, 60),
            (1, 61),
            (1, 62),
            (1, 63),
            (1, 64),
            (1, 65),
            (1, 66),
            (1, 67),
            (1, 68),
            (1, 69),
            (1, 70),
            (1, 71),
            (1, 72),
            (1, 73),
            (1, 74),
            (1, 75),
            (1, 76),
            (1, 77),
            (1, 78),
            (1, 79),
            (2, 30),
            (2, 31),
            (2, 32),
            (2, 33),
            (2, 34),
            (2, 35),
            (2, 36),
            (2, 37),
            (2, 38),
            (2, 39),
            (2, 40),
            (2, 41),
            (2, 42),
            (2, 43),
            (2, 44),
            (2, 45),
            (2, 46),
            (2, 47),
            (2, 48),
            (2, 49),
            (2, 50),
            (2, 51),
            (2, 52),
            (2, 53),
            (2, 54),
            (2, 55),
            (2, 56),
            (2, 57),
            (2, 58),
            (2, 59),
            (2, 60),
            (2, 61),
            (2, 62),
            (2, 63),
            (2, 64),
            (2, 65),
            (2, 66),
            (2, 67),
            (2, 68),
            (2, 69),
            (2, 70),
            (2, 71),
            (2, 72),
            (2, 73),
            (2, 74),
            (2, 75),
            (2, 76),
            (2, 77),
            (2, 78),
            (2, 79),
            (3, 33),
            (3, 34),
            (3, 35),
            (3, 36),
            (3, 37),
            (3, 38),
            (3, 39),
            (3, 40),
            (3, 41),
            (3, 42),
            (3, 43),
            (3, 44),
            (3, 45),
            (3, 46),
            (3, 47),
            (3, 48),
            (3, 49),
            (3, 50),
            (3, 51),
            (3, 52),
            (3, 53),
            (3, 54),
            (3, 55),
            (3, 56),
            (3, 57),
            (3, 58),
            (3, 59),
            (3, 60),
            (3, 61),
            (3, 62),
            (3, 63),
            (3, 64),
            (3, 65),
            (3, 66),
            (3, 67),
            (3, 68),
            (3, 69),
            (3, 70),
            (3, 71),
            (3, 72),
            (3, 73),
            (3, 74),
            (3, 75),
            (3, 76),
            (3, 77),
            (3, 78),
            (3, 79),
            (4, 35),
            (4, 36),
            (4, 37),
            (4, 38),
            (4, 39),
            (4, 40),
            (4, 41),
            (4, 42),
            (4, 43),
            (4, 44),
            (4, 45),
            (4, 46),
            (4, 47),
            (4, 48),
            (4, 49),
            (4, 50),
            (4, 51),
            (4, 52),
            (4, 53),
            (4, 54),
            (4, 55),
            (4, 56),
            (4, 57),
            (4, 58),
            (4, 59),
            (4, 60),
            (4, 61),
            (4, 62),
            (4, 63),
            (4, 64),
            (4, 65),
            (4, 66),
            (4, 67),
            (4, 68),
            (4, 69),
            (4, 70),
            (4, 71),
            (4, 72),
            (4, 73),
            (4, 74),
            (4, 75),
            (4, 76),
            (4, 77),
            (4, 78),
            (4, 79),
            (5, 38),
            (5, 39),
            (5, 40),
            (5, 41),
            (5, 42),
            (5, 43),
            (5, 44),
            (5, 45),
            (5, 46),
            (5, 47),
            (5, 48),
            (5, 49),
            (5, 50),
            (5, 51),
            (5, 52),
            (5, 53),
            (5, 54),
            (5, 55),
            (5, 56),
            (5, 57),
            (5, 58),
            (5, 59),
            (5, 60),
            (5, 61),
            (5, 62),
            (5, 63),
            (5, 64),
            (5, 65),
            (5, 66),
            (5, 67),
            (5, 68),
            (5, 69),
            (5, 70),
            (5, 71),
            (5, 72),
            (5, 73),
            (5, 74),
            (5, 75),
            (5, 76),
            (5, 77),
            (5, 78),
            (5, 79),
            (6, 41),
            (6, 42),
            (6, 43),
            (6, 44),
            (6, 45),
            (6, 46),
            (6, 47),
            (6, 48),
            (6, 49),
            (6, 50),
            (6, 51),
            (6, 52),
            (6, 53),
            (6, 54),
            (6, 55),
            (6, 56),
            (6, 57),
            (6, 58),
            (6, 59),
            (6, 60),
            (6, 61),
            (6, 62),
            (6, 63),
            (6, 64),
            (6, 65),
            (6, 66),
            (6, 67),
            (6, 68),
            (6, 69),
            (6, 70),
            (6, 71),
            (6, 72),
            (6, 73),
            (6, 74),
            (6, 75),
            (6, 76),
            (6, 77),
            (6, 78),
            (6, 79),
            (7, 43),
            (7, 44),
            (7, 45),
            (7, 46),
            (7, 47),
            (7, 48),
            (7, 49),
            (7, 50),
            (7, 51),
            (7, 52),
            (7, 53),
            (7, 54),
            (7, 55),
            (7, 56),
            (7, 57),
            (7, 58),
            (7, 59),
            (7, 60),
            (7, 61),
            (7, 62),
            (7, 63),
            (7, 64),
            (7, 65),
            (7, 66),
            (7, 67),
            (7, 68),
            (7, 69),
            (7, 70),
            (7, 71),
            (7, 72),
            (7, 73),
            (7, 74),
            (7, 75),
            (7, 76),
            (7, 77),
            (7, 78),
            (7, 79),
            (8, 46),
            (8, 47),
            (8, 48),
            (8, 49),
            (8, 50),
            (8, 51),
            (8, 52),
            (8, 53),
            (8, 54),
            (8, 55),
            (8, 56),
            (8, 57),
            (8, 58),
            (8, 59),
            (8, 60),
            (8, 61),
            (8, 62),
            (8, 63),
            (8, 64),
            (8, 65),
            (8, 66),
            (8, 67),
            (8, 68),
            (8, 69),
            (8, 70),
            (8, 71),
            (8, 72),
            (8, 73),
            (8, 74),
            (8, 75),
            (8, 76),
            (8, 77),
            (8, 78),
            (8, 79),
            (9, 49),
            (9, 50),
            (9, 51),
            (9, 52),
            (9, 53),
            (9, 54),
            (9, 55),
            (9, 56),
            (9, 57),
            (9, 58),
            (9, 59),
            (9, 60),
            (9, 61),
            (9, 62),
            (9, 63),
            (9, 64),
            (9, 65),
            (9, 66),
            (9, 67),
            (9, 68),
            (9, 69),
            (9, 70),
            (9, 71),
            (9, 72),
            (9, 73),
            (9, 74),
            (9, 75),
            (9, 76),
            (9, 77),
            (9, 78),
            (9, 79),
            (10, 52),
            (10, 53),
            (10, 54),
            (10, 55),
            (10, 56),
            (10, 57),
            (10, 58),
            (10, 59),
            (10, 60),
            (10, 61),
            (10, 62),
            (10, 63),
            (10, 64),
            (10, 65),
            (10, 66),
            (10, 67),
            (10, 68),
            (10, 69),
            (10, 70),
            (10, 71),
            (10, 72),
            (10, 73),
            (10, 74),
            (10, 75),
            (10, 76),
            (10, 77),
            (10, 78),
            (10, 79),
            (11, 54),
            (11, 55),
            (11, 56),
            (11, 57),
            (11, 58),
            (11, 59),
            (11, 60),
            (11, 61),
            (11, 62),
            (11, 63),
            (11, 64),
            (11, 65),
            (11, 66),
            (11, 67),
            (11, 68),
            (11, 69),
            (11, 70),
            (11, 71),
            (11, 72),
            (11, 73),
            (11, 74),
            (11, 75),
            (11, 76),
            (11, 77),
            (11, 78),
            (11, 79),
            (12, 57),
            (12, 58),
            (12, 59),
            (12, 60),
            (12, 61),
            (12, 62),
            (12, 63),
            (12, 64),
            (12, 65),
            (12, 66),
            (12, 67),
            (12, 68),
            (12, 69),
            (12, 70),
            (12, 71),
            (12, 72),
            (12, 73),
            (12, 74),
            (12, 75),
            (12, 76),
            (12, 77),
            (12, 78),
            (12, 79),
            (13, 60),
            (13, 61),
            (13, 62),
            (13, 63),
            (13, 64),
            (13, 65),
            (13, 66),
            (13, 67),
            (13, 68),
            (13, 69),
            (13, 70),
            (13, 71),
            (13, 72),
            (13, 73),
            (13, 74),
            (13, 75),
            (13, 76),
            (13, 77),
            (13, 78),
            (13, 79),
            (14, 62),
            (14, 63),
            (14, 64),
            (14, 65),
            (14, 66),
            (14, 67),
            (14, 68),
            (14, 69),
            (14, 70),
            (14, 71),
            (14, 72),
            (14, 73),
            (14, 74),
            (14, 75),
            (14, 76),
            (14, 77),
            (14, 78),
            (14, 79),
            (15, 65),
            (15, 66),
            (15, 67),
            (15, 68),
            (15, 69),
            (15, 70),
            (15, 71),
            (15, 72),
            (15, 73),
            (15, 74),
            (15, 75),
            (15, 76),
            (15, 77),
            (15, 78),
            (15, 79),
            (16, 68),
            (16, 69),
            (16, 70),
            (16, 71),
            (16, 72),
            (16, 73),
            (16, 74),
            (16, 75),
            (16, 76),
            (16, 77),
            (16, 78),
            (16, 79),
            (17, 71),
            (17, 72),
            (17, 73),
            (17, 74),
            (17, 75),
            (17, 76),
            (17, 77),
            (17, 78),
            (17, 79),
            (18, 73),
            (18, 74),
            (18, 75),
            (18, 76),
            (18, 77),
            (18, 78),
            (18, 79),
            (19, 76),
            (19, 77),
            (19, 78),
            (19, 79),
            (20, 79),
        }:
            return 75
        return 36

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_2_output, num_attn_0_0_output):
        key = (num_attn_0_2_output, num_attn_0_0_output)
        return 41

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(token, position):
        if token in {"<s>", "(", "{", "}"}:
            return position == 6
        elif token in {")"}:
            return position == 8

    attn_1_0_pattern = select_closest(positions, tokens, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_2_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(token, mlp_0_1_output):
        if token in {"(", ")", "{", "}"}:
            return mlp_0_1_output == 5
        elif token in {"<s>"}:
            return mlp_0_1_output == 6

    attn_1_1_pattern = select_closest(mlp_0_1_outputs, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_0_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")", "}"}:
            return position == 2
        elif token in {"<s>", "{"}:
            return position == 6

    attn_1_2_pattern = select_closest(positions, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, mlp_0_0_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(token, position):
        if token in {"<s>", "("}:
            return position == 7
        elif token in {")", "}"}:
            return position == 5
        elif token in {"{"}:
            return position == 6

    attn_1_3_pattern = select_closest(positions, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_0_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_3_output, mlp_0_0_output):
        if attn_0_3_output in {"<s>", ")", "{", "(", "}"}:
            return mlp_0_0_output == 27

    num_attn_1_0_pattern = select(mlp_0_0_outputs, attn_0_3_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_0_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(num_mlp_0_2_output, mlp_0_2_output):
        if num_mlp_0_2_output in {0, 2, 54}:
            return mlp_0_2_output == 1
        elif num_mlp_0_2_output in {1}:
            return mlp_0_2_output == 0
        elif num_mlp_0_2_output in {65, 3, 5}:
            return mlp_0_2_output == 71
        elif num_mlp_0_2_output in {10, 4, 66}:
            return mlp_0_2_output == 11
        elif num_mlp_0_2_output in {14, 23, 6, 7}:
            return mlp_0_2_output == 69
        elif num_mlp_0_2_output in {8}:
            return mlp_0_2_output == 49
        elif num_mlp_0_2_output in {9, 18}:
            return mlp_0_2_output == 46
        elif num_mlp_0_2_output in {16, 64, 11, 31}:
            return mlp_0_2_output == 42
        elif num_mlp_0_2_output in {67, 43, 12}:
            return mlp_0_2_output == 25
        elif num_mlp_0_2_output in {34, 13}:
            return mlp_0_2_output == 54
        elif num_mlp_0_2_output in {15}:
            return mlp_0_2_output == 20
        elif num_mlp_0_2_output in {17}:
            return mlp_0_2_output == 79
        elif num_mlp_0_2_output in {19}:
            return mlp_0_2_output == 8
        elif num_mlp_0_2_output in {20, 76}:
            return mlp_0_2_output == 61
        elif num_mlp_0_2_output in {40, 74, 77, 78, 52, 21}:
            return mlp_0_2_output == 6
        elif num_mlp_0_2_output in {22}:
            return mlp_0_2_output == 63
        elif num_mlp_0_2_output in {24, 69}:
            return mlp_0_2_output == 30
        elif num_mlp_0_2_output in {25, 50}:
            return mlp_0_2_output == 45
        elif num_mlp_0_2_output in {26, 27, 60, 79}:
            return mlp_0_2_output == 70
        elif num_mlp_0_2_output in {72, 28, 46}:
            return mlp_0_2_output == 9
        elif num_mlp_0_2_output in {29, 55}:
            return mlp_0_2_output == 58
        elif num_mlp_0_2_output in {33, 36, 37, 30}:
            return mlp_0_2_output == 24
        elif num_mlp_0_2_output in {32, 73}:
            return mlp_0_2_output == 78
        elif num_mlp_0_2_output in {35}:
            return mlp_0_2_output == 43
        elif num_mlp_0_2_output in {38}:
            return mlp_0_2_output == 31
        elif num_mlp_0_2_output in {39}:
            return mlp_0_2_output == 52
        elif num_mlp_0_2_output in {41, 53}:
            return mlp_0_2_output == 44
        elif num_mlp_0_2_output in {56, 42}:
            return mlp_0_2_output == 15
        elif num_mlp_0_2_output in {44}:
            return mlp_0_2_output == 47
        elif num_mlp_0_2_output in {45}:
            return mlp_0_2_output == 33
        elif num_mlp_0_2_output in {47}:
            return mlp_0_2_output == 41
        elif num_mlp_0_2_output in {48, 75}:
            return mlp_0_2_output == 39
        elif num_mlp_0_2_output in {49}:
            return mlp_0_2_output == 62
        elif num_mlp_0_2_output in {51}:
            return mlp_0_2_output == 19
        elif num_mlp_0_2_output in {57}:
            return mlp_0_2_output == 64
        elif num_mlp_0_2_output in {58}:
            return mlp_0_2_output == 76
        elif num_mlp_0_2_output in {59}:
            return mlp_0_2_output == 59
        elif num_mlp_0_2_output in {61}:
            return mlp_0_2_output == 14
        elif num_mlp_0_2_output in {62}:
            return mlp_0_2_output == 16
        elif num_mlp_0_2_output in {63}:
            return mlp_0_2_output == 12
        elif num_mlp_0_2_output in {68}:
            return mlp_0_2_output == 13
        elif num_mlp_0_2_output in {70}:
            return mlp_0_2_output == 56
        elif num_mlp_0_2_output in {71}:
            return mlp_0_2_output == 77

    num_attn_1_1_pattern = select(
        mlp_0_2_outputs, num_mlp_0_2_outputs, num_predicate_1_1
    )
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_0_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(num_mlp_0_2_output, mlp_0_2_output):
        if num_mlp_0_2_output in {0, 32, 41, 10, 13, 49, 22, 59, 29}:
            return mlp_0_2_output == 11
        elif num_mlp_0_2_output in {65, 1, 7}:
            return mlp_0_2_output == 65
        elif num_mlp_0_2_output in {2, 47}:
            return mlp_0_2_output == 7
        elif num_mlp_0_2_output in {43, 3}:
            return mlp_0_2_output == 24
        elif num_mlp_0_2_output in {67, 4, 36, 21, 61}:
            return mlp_0_2_output == 47
        elif num_mlp_0_2_output in {64, 44, 5, 14}:
            return mlp_0_2_output == 16
        elif num_mlp_0_2_output in {77, 6}:
            return mlp_0_2_output == 26
        elif num_mlp_0_2_output in {8, 42}:
            return mlp_0_2_output == 39
        elif num_mlp_0_2_output in {9}:
            return mlp_0_2_output == 21
        elif num_mlp_0_2_output in {11}:
            return mlp_0_2_output == 42
        elif num_mlp_0_2_output in {12}:
            return mlp_0_2_output == 61
        elif num_mlp_0_2_output in {52, 15}:
            return mlp_0_2_output == 14
        elif num_mlp_0_2_output in {16}:
            return mlp_0_2_output == 1
        elif num_mlp_0_2_output in {17, 19, 51}:
            return mlp_0_2_output == 50
        elif num_mlp_0_2_output in {18, 54}:
            return mlp_0_2_output == 13
        elif num_mlp_0_2_output in {20, 30}:
            return mlp_0_2_output == 6
        elif num_mlp_0_2_output in {23}:
            return mlp_0_2_output == 56
        elif num_mlp_0_2_output in {24, 68}:
            return mlp_0_2_output == 52
        elif num_mlp_0_2_output in {25}:
            return mlp_0_2_output == 60
        elif num_mlp_0_2_output in {26}:
            return mlp_0_2_output == 77
        elif num_mlp_0_2_output in {27}:
            return mlp_0_2_output == 63
        elif num_mlp_0_2_output in {28}:
            return mlp_0_2_output == 15
        elif num_mlp_0_2_output in {55, 45, 31}:
            return mlp_0_2_output == 55
        elif num_mlp_0_2_output in {40, 33}:
            return mlp_0_2_output == 17
        elif num_mlp_0_2_output in {72, 34}:
            return mlp_0_2_output == 23
        elif num_mlp_0_2_output in {35}:
            return mlp_0_2_output == 62
        elif num_mlp_0_2_output in {37}:
            return mlp_0_2_output == 36
        elif num_mlp_0_2_output in {38}:
            return mlp_0_2_output == 64
        elif num_mlp_0_2_output in {39}:
            return mlp_0_2_output == 30
        elif num_mlp_0_2_output in {48, 69, 46}:
            return mlp_0_2_output == 29
        elif num_mlp_0_2_output in {50}:
            return mlp_0_2_output == 66
        elif num_mlp_0_2_output in {53, 71}:
            return mlp_0_2_output == 9
        elif num_mlp_0_2_output in {56}:
            return mlp_0_2_output == 76
        elif num_mlp_0_2_output in {57, 62, 63}:
            return mlp_0_2_output == 10
        elif num_mlp_0_2_output in {58, 79}:
            return mlp_0_2_output == 4
        elif num_mlp_0_2_output in {66, 60}:
            return mlp_0_2_output == 19
        elif num_mlp_0_2_output in {70}:
            return mlp_0_2_output == 45
        elif num_mlp_0_2_output in {73}:
            return mlp_0_2_output == 70
        elif num_mlp_0_2_output in {74}:
            return mlp_0_2_output == 44
        elif num_mlp_0_2_output in {75}:
            return mlp_0_2_output == 46
        elif num_mlp_0_2_output in {76}:
            return mlp_0_2_output == 54
        elif num_mlp_0_2_output in {78}:
            return mlp_0_2_output == 0

    num_attn_1_2_pattern = select(
        mlp_0_2_outputs, num_mlp_0_2_outputs, num_predicate_1_2
    )
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_0_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(mlp_0_1_output, mlp_0_2_output):
        if mlp_0_1_output in {0}:
            return mlp_0_2_output == 31
        elif mlp_0_1_output in {
            1,
            4,
            14,
            18,
            20,
            30,
            37,
            47,
            56,
            57,
            60,
            61,
            65,
            67,
            69,
            72,
            73,
            74,
            75,
            76,
        }:
            return mlp_0_2_output == 2
        elif mlp_0_1_output in {64, 33, 2, 5, 41, 10, 48, 54, 29, 63}:
            return mlp_0_2_output == 27
        elif mlp_0_1_output in {32, 3, 36, 39, 11, 13, 21, 24, 25}:
            return mlp_0_2_output == 68
        elif mlp_0_1_output in {38, 6, 71, 12, 52, 55, 26}:
            return mlp_0_2_output == 24
        elif mlp_0_1_output in {17, 62, 7}:
            return mlp_0_2_output == 63
        elif mlp_0_1_output in {
            34,
            35,
            68,
            8,
            43,
            44,
            45,
            46,
            77,
            78,
            49,
            50,
            19,
            22,
            23,
            27,
            28,
            31,
        }:
            return mlp_0_2_output == 74
        elif mlp_0_1_output in {9}:
            return mlp_0_2_output == 21
        elif mlp_0_1_output in {40, 15}:
            return mlp_0_2_output == 41
        elif mlp_0_1_output in {16}:
            return mlp_0_2_output == 67
        elif mlp_0_1_output in {42}:
            return mlp_0_2_output == 55
        elif mlp_0_1_output in {51}:
            return mlp_0_2_output == 16
        elif mlp_0_1_output in {53}:
            return mlp_0_2_output == 30
        elif mlp_0_1_output in {58}:
            return mlp_0_2_output == 70
        elif mlp_0_1_output in {59}:
            return mlp_0_2_output == 19
        elif mlp_0_1_output in {66}:
            return mlp_0_2_output == 49
        elif mlp_0_1_output in {70}:
            return mlp_0_2_output == 47
        elif mlp_0_1_output in {79}:
            return mlp_0_2_output == 28

    num_attn_1_3_pattern = select(mlp_0_2_outputs, mlp_0_1_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_0_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(token, position):
        key = (token, position)
        if key in {
            ("(", 2),
            (")", 0),
            (")", 1),
            (")", 2),
            (")", 3),
            (")", 4),
            (")", 5),
            (")", 6),
            (")", 8),
            (")", 12),
            (")", 17),
            (")", 28),
            (")", 30),
            (")", 34),
            (")", 36),
            (")", 38),
            (")", 40),
            (")", 41),
            (")", 42),
            (")", 43),
            (")", 44),
            (")", 45),
            (")", 46),
            (")", 47),
            (")", 48),
            (")", 49),
            (")", 50),
            (")", 51),
            (")", 52),
            (")", 53),
            (")", 54),
            (")", 55),
            (")", 56),
            (")", 57),
            (")", 58),
            (")", 59),
            (")", 60),
            (")", 61),
            (")", 62),
            (")", 63),
            (")", 64),
            (")", 65),
            (")", 66),
            (")", 67),
            (")", 68),
            (")", 69),
            (")", 70),
            (")", 71),
            (")", 72),
            (")", 73),
            (")", 74),
            (")", 75),
            (")", 76),
            (")", 77),
            (")", 78),
            (")", 79),
            ("<s>", 0),
            ("<s>", 2),
            ("<s>", 3),
            ("<s>", 5),
            ("<s>", 6),
            ("<s>", 12),
            ("<s>", 17),
            ("<s>", 28),
            ("<s>", 34),
            ("<s>", 36),
            ("<s>", 38),
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
            ("<s>", 50),
            ("<s>", 51),
            ("<s>", 52),
            ("<s>", 53),
            ("<s>", 54),
            ("<s>", 55),
            ("<s>", 56),
            ("<s>", 57),
            ("<s>", 58),
            ("<s>", 59),
            ("<s>", 60),
            ("<s>", 61),
            ("<s>", 62),
            ("<s>", 63),
            ("<s>", 64),
            ("<s>", 65),
            ("<s>", 66),
            ("<s>", 67),
            ("<s>", 68),
            ("<s>", 69),
            ("<s>", 70),
            ("<s>", 71),
            ("<s>", 72),
            ("<s>", 73),
            ("<s>", 74),
            ("<s>", 75),
            ("<s>", 76),
            ("<s>", 77),
            ("<s>", 78),
            ("<s>", 79),
            ("}", 0),
            ("}", 1),
            ("}", 2),
            ("}", 3),
            ("}", 4),
            ("}", 5),
            ("}", 6),
            ("}", 8),
            ("}", 12),
            ("}", 17),
            ("}", 28),
            ("}", 30),
            ("}", 34),
            ("}", 36),
            ("}", 38),
            ("}", 40),
            ("}", 41),
            ("}", 42),
            ("}", 43),
            ("}", 44),
            ("}", 45),
            ("}", 46),
            ("}", 47),
            ("}", 48),
            ("}", 49),
            ("}", 50),
            ("}", 51),
            ("}", 52),
            ("}", 53),
            ("}", 54),
            ("}", 55),
            ("}", 56),
            ("}", 57),
            ("}", 58),
            ("}", 59),
            ("}", 60),
            ("}", 61),
            ("}", 62),
            ("}", 63),
            ("}", 64),
            ("}", 65),
            ("}", 66),
            ("}", 67),
            ("}", 68),
            ("}", 69),
            ("}", 70),
            ("}", 71),
            ("}", 72),
            ("}", 73),
            ("}", 74),
            ("}", 75),
            ("}", 76),
            ("}", 77),
            ("}", 78),
            ("}", 79),
        }:
            return 40
        elif key in {
            ("(", 0),
            ("(", 7),
            ("(", 8),
            ("(", 14),
            ("(", 15),
            ("(", 16),
            ("(", 18),
            ("(", 19),
            ("(", 24),
            ("(", 27),
            ("(", 29),
            ("(", 30),
            ("(", 31),
            ("(", 35),
            ("(", 36),
            ("(", 37),
            ("(", 39),
            ("(", 40),
            ("(", 41),
            ("(", 42),
            ("(", 43),
            ("(", 44),
            ("(", 45),
            ("(", 46),
            ("(", 47),
            ("(", 48),
            ("(", 49),
            ("(", 50),
            ("(", 51),
            ("(", 52),
            ("(", 53),
            ("(", 54),
            ("(", 55),
            ("(", 56),
            ("(", 57),
            ("(", 58),
            ("(", 59),
            ("(", 60),
            ("(", 61),
            ("(", 62),
            ("(", 63),
            ("(", 64),
            ("(", 65),
            ("(", 66),
            ("(", 67),
            ("(", 68),
            ("(", 69),
            ("(", 70),
            ("(", 71),
            ("(", 72),
            ("(", 73),
            ("(", 74),
            ("(", 75),
            ("(", 76),
            ("(", 77),
            ("(", 78),
            ("(", 79),
            ("<s>", 7),
            ("<s>", 16),
            ("{", 0),
            ("{", 2),
            ("{", 5),
            ("{", 6),
            ("{", 7),
            ("{", 8),
            ("{", 9),
            ("{", 10),
            ("{", 11),
            ("{", 12),
            ("{", 13),
            ("{", 14),
            ("{", 15),
            ("{", 16),
            ("{", 17),
            ("{", 18),
            ("{", 19),
            ("{", 20),
            ("{", 21),
            ("{", 22),
            ("{", 23),
            ("{", 24),
            ("{", 25),
            ("{", 27),
            ("{", 28),
            ("{", 29),
            ("{", 30),
            ("{", 31),
            ("{", 34),
            ("{", 35),
            ("{", 36),
            ("{", 37),
            ("{", 38),
            ("{", 39),
            ("{", 40),
            ("{", 41),
            ("{", 42),
            ("{", 43),
            ("{", 44),
            ("{", 45),
            ("{", 46),
            ("{", 47),
            ("{", 48),
            ("{", 49),
            ("{", 50),
            ("{", 51),
            ("{", 52),
            ("{", 53),
            ("{", 54),
            ("{", 55),
            ("{", 56),
            ("{", 57),
            ("{", 58),
            ("{", 59),
            ("{", 60),
            ("{", 61),
            ("{", 62),
            ("{", 63),
            ("{", 64),
            ("{", 65),
            ("{", 66),
            ("{", 67),
            ("{", 68),
            ("{", 69),
            ("{", 70),
            ("{", 71),
            ("{", 72),
            ("{", 73),
            ("{", 74),
            ("{", 75),
            ("{", 76),
            ("{", 77),
            ("{", 78),
            ("{", 79),
        }:
            return 25
        elif key in {
            ("(", 9),
            ("(", 10),
            ("(", 13),
            ("(", 17),
            ("(", 21),
            ("(", 22),
            ("(", 25),
            ("(", 28),
            ("(", 33),
            ("(", 34),
            (")", 10),
            (")", 14),
            (")", 20),
            (")", 22),
            (")", 35),
            ("<s>", 14),
            ("<s>", 20),
            ("<s>", 22),
            ("<s>", 25),
            ("<s>", 35),
            ("{", 33),
            ("}", 14),
            ("}", 20),
            ("}", 22),
            ("}", 35),
        }:
            return 4
        elif key in {
            ("(", 5),
            ("(", 6),
            ("(", 11),
            ("(", 12),
            ("(", 20),
            ("(", 32),
            ("(", 38),
            ("<s>", 1),
            ("{", 32),
        }:
            return 41
        elif key in {
            ("(", 23),
            (")", 7),
            (")", 23),
            ("<s>", 10),
            ("<s>", 23),
            ("}", 7),
            ("}", 10),
            ("}", 23),
        }:
            return 18
        elif key in {("(", 4), ("<s>", 4), ("{", 4)}:
            return 73
        elif key in {("(", 1), ("{", 1)}:
            return 38
        elif key in {("(", 3), ("{", 3)}:
            return 78
        return 77

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(tokens, positions)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_2_output, attn_0_3_output):
        key = (attn_0_2_output, attn_0_3_output)
        if key in {(")", "<s>"), ("<s>", "<s>")}:
            return 23
        elif key in {
            ("(", "("),
            ("(", ")"),
            ("(", "{"),
            ("(", "}"),
            (")", "{"),
            (")", "}"),
            ("{", "("),
            ("{", ")"),
            ("}", ")"),
        }:
            return 5
        elif key in {("(", "<s>")}:
            return 78
        elif key in {("}", "(")}:
            return 61
        elif key in {("}", "<s>")}:
            return 79
        return 7

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_3_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_1_1_output, attn_1_0_output):
        key = (attn_1_1_output, attn_1_0_output)
        if key in {
            (0, ")"),
            (0, "{"),
            (1, ")"),
            (1, "{"),
            (2, "{"),
            (3, "{"),
            (4, ")"),
            (4, "{"),
            (5, "{"),
            (6, ")"),
            (6, "{"),
            (7, ")"),
            (7, "{"),
            (8, ")"),
            (8, "{"),
            (9, ")"),
            (9, "{"),
            (10, ")"),
            (10, "{"),
            (11, ")"),
            (11, "{"),
            (12, ")"),
            (12, "{"),
            (13, "{"),
            (14, ")"),
            (14, "{"),
            (15, ")"),
            (15, "{"),
            (16, ")"),
            (16, "{"),
            (18, "{"),
            (19, ")"),
            (19, "{"),
            (20, ")"),
            (20, "{"),
            (21, ")"),
            (21, "{"),
            (22, ")"),
            (22, "{"),
            (23, ")"),
            (23, "{"),
            (24, ")"),
            (24, "{"),
            (25, ")"),
            (25, "{"),
            (26, ")"),
            (26, "{"),
            (27, ")"),
            (27, "{"),
            (28, ")"),
            (28, "{"),
            (29, ")"),
            (29, "{"),
            (30, ")"),
            (30, "{"),
            (32, ")"),
            (32, "{"),
            (33, ")"),
            (33, "{"),
            (34, ")"),
            (34, "{"),
            (35, ")"),
            (35, "{"),
            (36, "{"),
            (37, ")"),
            (37, "{"),
            (38, ")"),
            (38, "{"),
            (39, ")"),
            (39, "{"),
            (41, ")"),
            (41, "{"),
            (42, ")"),
            (42, "{"),
            (43, ")"),
            (43, "{"),
            (44, ")"),
            (44, "{"),
            (45, ")"),
            (45, "{"),
            (46, ")"),
            (46, "{"),
            (47, ")"),
            (47, "{"),
            (48, ")"),
            (48, "{"),
            (49, ")"),
            (49, "{"),
            (50, ")"),
            (50, "{"),
            (51, "{"),
            (52, ")"),
            (52, "{"),
            (53, ")"),
            (53, "{"),
            (54, ")"),
            (54, "{"),
            (55, ")"),
            (55, "{"),
            (56, ")"),
            (56, "{"),
            (57, ")"),
            (57, "{"),
            (58, ")"),
            (58, "{"),
            (59, ")"),
            (59, "{"),
            (60, ")"),
            (60, "{"),
            (61, ")"),
            (61, "{"),
            (62, ")"),
            (62, "{"),
            (63, ")"),
            (63, "{"),
            (64, ")"),
            (64, "{"),
            (65, ")"),
            (65, "{"),
            (66, ")"),
            (66, "{"),
            (67, ")"),
            (67, "{"),
            (69, ")"),
            (69, "{"),
            (70, ")"),
            (70, "{"),
            (71, ")"),
            (71, "{"),
            (72, ")"),
            (72, "{"),
            (73, ")"),
            (73, "{"),
            (74, ")"),
            (74, "<s>"),
            (74, "{"),
            (75, ")"),
            (75, "{"),
            (76, ")"),
            (76, "{"),
            (77, ")"),
            (77, "{"),
            (78, ")"),
            (78, "{"),
            (79, ")"),
            (79, "{"),
        }:
            return 12
        elif key in {(2, "("), (17, "("), (17, ")"), (17, "{"), (17, "}"), (27, "(")}:
            return 16
        elif key in {(31, "("), (31, ")"), (31, "<s>"), (31, "{"), (46, "(")}:
            return 22
        elif key in {(17, "<s>"), (33, "<s>")}:
            return 30
        elif key in {(2, ")"), (13, ")"), (18, ")"), (36, ")")}:
            return 66
        elif key in {(40, "{")}:
            return 19
        elif key in {(51, ")")}:
            return 8
        return 72

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_1_0_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_0_1_output, attn_1_1_output):
        key = (attn_0_1_output, attn_1_1_output)
        if key in {
            ("(", 2),
            ("(", 3),
            ("(", 5),
            ("(", 31),
            ("(", 33),
            ("(", 40),
            ("(", 68),
            ("(", 74),
            (")", 2),
            (")", 3),
            (")", 5),
            (")", 13),
            (")", 17),
            (")", 31),
            (")", 33),
            (")", 40),
            (")", 46),
            (")", 48),
            (")", 63),
            (")", 68),
            (")", 74),
            (")", 75),
            ("{", 2),
            ("{", 3),
            ("{", 5),
            ("{", 31),
            ("{", 33),
            ("{", 40),
            ("{", 68),
            ("{", 74),
            ("}", 0),
            ("}", 1),
            ("}", 2),
            ("}", 3),
            ("}", 5),
            ("}", 6),
            ("}", 7),
            ("}", 8),
            ("}", 9),
            ("}", 10),
            ("}", 11),
            ("}", 12),
            ("}", 13),
            ("}", 14),
            ("}", 15),
            ("}", 16),
            ("}", 17),
            ("}", 18),
            ("}", 19),
            ("}", 20),
            ("}", 21),
            ("}", 22),
            ("}", 23),
            ("}", 24),
            ("}", 25),
            ("}", 26),
            ("}", 28),
            ("}", 29),
            ("}", 30),
            ("}", 31),
            ("}", 33),
            ("}", 34),
            ("}", 35),
            ("}", 36),
            ("}", 37),
            ("}", 38),
            ("}", 39),
            ("}", 40),
            ("}", 41),
            ("}", 42),
            ("}", 43),
            ("}", 44),
            ("}", 45),
            ("}", 46),
            ("}", 47),
            ("}", 48),
            ("}", 49),
            ("}", 50),
            ("}", 51),
            ("}", 52),
            ("}", 53),
            ("}", 54),
            ("}", 55),
            ("}", 56),
            ("}", 57),
            ("}", 58),
            ("}", 59),
            ("}", 60),
            ("}", 61),
            ("}", 62),
            ("}", 63),
            ("}", 64),
            ("}", 65),
            ("}", 66),
            ("}", 67),
            ("}", 68),
            ("}", 69),
            ("}", 70),
            ("}", 71),
            ("}", 72),
            ("}", 73),
            ("}", 74),
            ("}", 75),
            ("}", 76),
            ("}", 77),
            ("}", 78),
            ("}", 79),
        }:
            return 33
        elif key in {
            ("(", 27),
            (")", 27),
            ("<s>", 0),
            ("<s>", 2),
            ("<s>", 3),
            ("<s>", 5),
            ("<s>", 6),
            ("<s>", 7),
            ("<s>", 8),
            ("<s>", 9),
            ("<s>", 11),
            ("<s>", 12),
            ("<s>", 13),
            ("<s>", 14),
            ("<s>", 15),
            ("<s>", 16),
            ("<s>", 18),
            ("<s>", 19),
            ("<s>", 20),
            ("<s>", 22),
            ("<s>", 23),
            ("<s>", 24),
            ("<s>", 25),
            ("<s>", 27),
            ("<s>", 29),
            ("<s>", 30),
            ("<s>", 32),
            ("<s>", 34),
            ("<s>", 35),
            ("<s>", 37),
            ("<s>", 38),
            ("<s>", 40),
            ("<s>", 41),
            ("<s>", 42),
            ("<s>", 43),
            ("<s>", 44),
            ("<s>", 47),
            ("<s>", 48),
            ("<s>", 50),
            ("<s>", 52),
            ("<s>", 53),
            ("<s>", 54),
            ("<s>", 56),
            ("<s>", 57),
            ("<s>", 58),
            ("<s>", 59),
            ("<s>", 60),
            ("<s>", 61),
            ("<s>", 62),
            ("<s>", 63),
            ("<s>", 64),
            ("<s>", 65),
            ("<s>", 66),
            ("<s>", 67),
            ("<s>", 68),
            ("<s>", 69),
            ("<s>", 71),
            ("<s>", 73),
            ("<s>", 74),
            ("<s>", 76),
            ("<s>", 77),
            ("<s>", 78),
            ("{", 27),
            ("}", 27),
        }:
            return 47
        elif key in {
            ("(", 17),
            ("<s>", 17),
            ("{", 0),
            ("{", 14),
            ("{", 17),
            ("{", 24),
            ("{", 32),
            ("{", 61),
        }:
            return 74
        elif key in {
            (")", 6),
            (")", 7),
            (")", 8),
            (")", 35),
            (")", 37),
            (")", 47),
            (")", 53),
            (")", 64),
            (")", 67),
            (")", 70),
            (")", 71),
            (")", 72),
            (")", 77),
            ("}", 32),
        }:
            return 62
        return 70

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_1_1_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_0_output, num_attn_1_2_output):
        key = (num_attn_1_0_output, num_attn_1_2_output)
        return 78

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_2_output, num_attn_1_1_output):
        key = (num_attn_1_2_output, num_attn_1_1_output)
        return 10

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_2_output, num_attn_1_0_output):
        key = (num_attn_1_2_output, num_attn_1_0_output)
        if key in {
            (0, 52),
            (0, 53),
            (0, 54),
            (0, 55),
            (0, 56),
            (0, 57),
            (0, 58),
            (0, 59),
            (0, 60),
            (0, 61),
            (0, 62),
            (0, 63),
            (0, 64),
            (0, 65),
            (0, 66),
            (0, 67),
            (0, 68),
            (0, 69),
            (0, 70),
            (0, 71),
            (0, 72),
            (0, 73),
            (0, 74),
            (0, 75),
            (0, 76),
            (0, 77),
            (0, 78),
            (0, 79),
            (0, 80),
            (0, 81),
            (0, 82),
            (0, 83),
            (0, 84),
            (0, 85),
            (0, 86),
            (0, 87),
            (0, 88),
            (0, 89),
            (0, 90),
            (0, 91),
            (0, 92),
            (0, 93),
            (0, 94),
            (0, 95),
            (0, 96),
            (0, 97),
            (0, 98),
            (0, 99),
            (0, 100),
            (0, 101),
            (0, 102),
            (0, 103),
            (0, 104),
            (0, 105),
            (0, 106),
            (0, 107),
            (0, 108),
            (0, 109),
            (0, 110),
            (0, 111),
            (0, 112),
            (0, 113),
            (0, 114),
            (0, 115),
            (0, 116),
            (0, 117),
            (0, 118),
            (0, 119),
            (0, 120),
            (0, 121),
            (0, 122),
            (0, 123),
            (0, 124),
            (0, 125),
            (0, 126),
            (0, 127),
            (0, 128),
            (0, 129),
            (0, 130),
            (0, 131),
            (0, 132),
            (0, 133),
            (0, 134),
            (0, 135),
            (0, 136),
            (0, 137),
            (0, 138),
            (0, 139),
            (0, 140),
            (0, 141),
            (0, 142),
            (0, 143),
            (0, 144),
            (0, 145),
            (0, 146),
            (0, 147),
            (0, 148),
            (0, 149),
            (0, 150),
            (0, 151),
            (0, 152),
            (0, 153),
            (0, 154),
            (0, 155),
            (0, 156),
            (0, 157),
            (0, 158),
            (0, 159),
            (1, 73),
            (1, 74),
            (1, 75),
            (1, 76),
            (1, 77),
            (1, 78),
            (1, 79),
            (1, 80),
            (1, 81),
            (1, 82),
            (1, 83),
            (1, 84),
            (1, 85),
            (1, 86),
            (1, 87),
            (1, 88),
            (1, 89),
            (1, 90),
            (1, 91),
            (1, 92),
            (1, 93),
            (1, 94),
            (1, 95),
            (1, 96),
            (1, 97),
            (1, 98),
            (1, 99),
            (1, 100),
            (1, 101),
            (1, 102),
            (1, 103),
            (1, 104),
            (1, 105),
            (1, 106),
            (1, 107),
            (1, 108),
            (1, 109),
            (1, 110),
            (1, 111),
            (1, 112),
            (1, 113),
            (1, 114),
            (1, 115),
            (1, 116),
            (1, 117),
            (1, 118),
            (1, 119),
            (1, 120),
            (1, 121),
            (1, 122),
            (1, 123),
            (1, 124),
            (1, 125),
            (1, 126),
            (1, 127),
            (1, 128),
            (1, 129),
            (1, 130),
            (1, 131),
            (1, 132),
            (1, 133),
            (1, 134),
            (1, 135),
            (1, 136),
            (1, 137),
            (1, 138),
            (1, 139),
            (1, 140),
            (1, 141),
            (1, 142),
            (1, 143),
            (1, 144),
            (1, 145),
            (1, 146),
            (1, 147),
            (1, 148),
            (1, 149),
            (1, 150),
            (1, 151),
            (1, 152),
            (1, 153),
            (1, 154),
            (1, 155),
            (1, 156),
            (1, 157),
            (1, 158),
            (1, 159),
            (2, 94),
            (2, 95),
            (2, 96),
            (2, 97),
            (2, 98),
            (2, 99),
            (2, 100),
            (2, 101),
            (2, 102),
            (2, 103),
            (2, 104),
            (2, 105),
            (2, 106),
            (2, 107),
            (2, 108),
            (2, 109),
            (2, 110),
            (2, 111),
            (2, 112),
            (2, 113),
            (2, 114),
            (2, 115),
            (2, 116),
            (2, 117),
            (2, 118),
            (2, 119),
            (2, 120),
            (2, 121),
            (2, 122),
            (2, 123),
            (2, 124),
            (2, 125),
            (2, 126),
            (2, 127),
            (2, 128),
            (2, 129),
            (2, 130),
            (2, 131),
            (2, 132),
            (2, 133),
            (2, 134),
            (2, 135),
            (2, 136),
            (2, 137),
            (2, 138),
            (2, 139),
            (2, 140),
            (2, 141),
            (2, 142),
            (2, 143),
            (2, 144),
            (2, 145),
            (2, 146),
            (2, 147),
            (2, 148),
            (2, 149),
            (2, 150),
            (2, 151),
            (2, 152),
            (2, 153),
            (2, 154),
            (2, 155),
            (2, 156),
            (2, 157),
            (2, 158),
            (2, 159),
            (3, 114),
            (3, 115),
            (3, 116),
            (3, 117),
            (3, 118),
            (3, 119),
            (3, 120),
            (3, 121),
            (3, 122),
            (3, 123),
            (3, 124),
            (3, 125),
            (3, 126),
            (3, 127),
            (3, 128),
            (3, 129),
            (3, 130),
            (3, 131),
            (3, 132),
            (3, 133),
            (3, 134),
            (3, 135),
            (3, 136),
            (3, 137),
            (3, 138),
            (3, 139),
            (3, 140),
            (3, 141),
            (3, 142),
            (3, 143),
            (3, 144),
            (3, 145),
            (3, 146),
            (3, 147),
            (3, 148),
            (3, 149),
            (3, 150),
            (3, 151),
            (3, 152),
            (3, 153),
            (3, 154),
            (3, 155),
            (3, 156),
            (3, 157),
            (3, 158),
            (3, 159),
            (4, 135),
            (4, 136),
            (4, 137),
            (4, 138),
            (4, 139),
            (4, 140),
            (4, 141),
            (4, 142),
            (4, 143),
            (4, 144),
            (4, 145),
            (4, 146),
            (4, 147),
            (4, 148),
            (4, 149),
            (4, 150),
            (4, 151),
            (4, 152),
            (4, 153),
            (4, 154),
            (4, 155),
            (4, 156),
            (4, 157),
            (4, 158),
            (4, 159),
            (5, 156),
            (5, 157),
            (5, 158),
            (5, 159),
        }:
            return 41
        return 69

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_0_0_output, num_attn_0_3_output):
        key = (num_attn_0_0_output, num_attn_0_3_output)
        return 79

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_2_output, position):
        if attn_0_2_output in {"(", "{"}:
            return position == 14
        elif attn_0_2_output in {")"}:
            return position == 10
        elif attn_0_2_output in {"<s>"}:
            return position == 3
        elif attn_0_2_output in {"}"}:
            return position == 9

    attn_2_0_pattern = select_closest(positions, attn_0_2_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(token, position):
        if token in {"(", "{"}:
            return position == 3
        elif token in {")"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 9
        elif token in {"}"}:
            return position == 7

    attn_2_1_pattern = select_closest(positions, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_1_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_3_output, position):
        if attn_0_3_output in {"("}:
            return position == 14
        elif attn_0_3_output in {")"}:
            return position == 9
        elif attn_0_3_output in {"<s>"}:
            return position == 2
        elif attn_0_3_output in {"{"}:
            return position == 15
        elif attn_0_3_output in {"}"}:
            return position == 10

    attn_2_2_pattern = select_closest(positions, attn_0_3_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_2_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(token, position):
        if token in {"(", "{"}:
            return position == 3
        elif token in {")", "}"}:
            return position == 10
        elif token in {"<s>"}:
            return position == 9

    attn_2_3_pattern = select_closest(positions, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_2_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_0_2_output, attn_0_1_output):
        if attn_0_2_output in {"<s>", ")", "{", "(", "}"}:
            return attn_0_1_output == ""

    num_attn_2_0_pattern = select(attn_0_1_outputs, attn_0_2_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_1_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(num_mlp_1_0_output, mlp_0_3_output):
        if num_mlp_1_0_output in {
            0,
            2,
            5,
            6,
            7,
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
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
        }:
            return mlp_0_3_output == 4
        elif num_mlp_1_0_output in {1}:
            return mlp_0_3_output == 51
        elif num_mlp_1_0_output in {3}:
            return mlp_0_3_output == 47
        elif num_mlp_1_0_output in {4}:
            return mlp_0_3_output == 9
        elif num_mlp_1_0_output in {13}:
            return mlp_0_3_output == 33
        elif num_mlp_1_0_output in {19}:
            return mlp_0_3_output == 24
        elif num_mlp_1_0_output in {26}:
            return mlp_0_3_output == 29
        elif num_mlp_1_0_output in {50}:
            return mlp_0_3_output == 52
        elif num_mlp_1_0_output in {68}:
            return mlp_0_3_output == 66

    num_attn_2_1_pattern = select(
        mlp_0_3_outputs, num_mlp_1_0_outputs, num_predicate_2_1
    )
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_2_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_0_output, mlp_0_1_output):
        if attn_1_0_output in {"(", "{"}:
            return mlp_0_1_output == 69
        elif attn_1_0_output in {")"}:
            return mlp_0_1_output == 41
        elif attn_1_0_output in {"<s>"}:
            return mlp_0_1_output == 74
        elif attn_1_0_output in {"}"}:
            return mlp_0_1_output == 30

    num_attn_2_2_pattern = select(mlp_0_1_outputs, attn_1_0_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_3_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(mlp_1_3_output, mlp_0_0_output):
        if mlp_1_3_output in {0}:
            return mlp_0_0_output == 7
        elif mlp_1_3_output in {
            1,
            4,
            5,
            6,
            7,
            9,
            10,
            11,
            14,
            16,
            17,
            18,
            21,
            24,
            26,
            27,
            28,
            29,
            30,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            45,
            47,
            48,
            49,
            52,
            54,
            57,
            60,
            62,
            64,
            65,
            70,
            73,
            74,
            76,
            78,
        }:
            return mlp_0_0_output == 33
        elif mlp_1_3_output in {2}:
            return mlp_0_0_output == 46
        elif mlp_1_3_output in {3, 68, 71, 75, 12, 77, 20, 23}:
            return mlp_0_0_output == 40
        elif mlp_1_3_output in {67, 69, 8, 41, 72, 43, 46, 79, 53, 25, 58, 59, 61}:
            return mlp_0_0_output == 17
        elif mlp_1_3_output in {13}:
            return mlp_0_0_output == 39
        elif mlp_1_3_output in {15}:
            return mlp_0_0_output == 2
        elif mlp_1_3_output in {19}:
            return mlp_0_0_output == 49
        elif mlp_1_3_output in {32, 66, 42, 50, 22, 63}:
            return mlp_0_0_output == 31
        elif mlp_1_3_output in {31}:
            return mlp_0_0_output == 16
        elif mlp_1_3_output in {44}:
            return mlp_0_0_output == 32
        elif mlp_1_3_output in {51}:
            return mlp_0_0_output == 65
        elif mlp_1_3_output in {55}:
            return mlp_0_0_output == 70
        elif mlp_1_3_output in {56}:
            return mlp_0_0_output == 25

    num_attn_2_3_pattern = select(mlp_0_0_outputs, mlp_1_3_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_3_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_1_output, attn_2_2_output):
        key = (attn_2_1_output, attn_2_2_output)
        if key in {
            (0, 32),
            (1, 32),
            (15, 1),
            (15, 5),
            (15, 13),
            (15, 18),
            (15, 40),
            (17, 32),
            (30, 13),
            (30, 18),
            (34, 1),
            (34, 5),
            (34, 13),
            (34, 18),
            (37, 0),
            (37, 1),
            (37, 3),
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
            (37, 28),
            (37, 29),
            (37, 30),
            (37, 31),
            (37, 32),
            (37, 33),
            (37, 34),
            (37, 35),
            (37, 36),
            (37, 37),
            (37, 38),
            (37, 39),
            (37, 41),
            (37, 42),
            (37, 43),
            (37, 44),
            (37, 45),
            (37, 46),
            (37, 47),
            (37, 48),
            (37, 49),
            (37, 50),
            (37, 51),
            (37, 52),
            (37, 53),
            (37, 54),
            (37, 55),
            (37, 56),
            (37, 57),
            (37, 58),
            (37, 59),
            (37, 60),
            (37, 61),
            (37, 62),
            (37, 63),
            (37, 64),
            (37, 65),
            (37, 66),
            (37, 67),
            (37, 68),
            (37, 69),
            (37, 70),
            (37, 71),
            (37, 72),
            (37, 73),
            (37, 74),
            (37, 75),
            (37, 76),
            (37, 77),
            (37, 78),
            (37, 79),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
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
            (39, 28),
            (39, 29),
            (39, 30),
            (39, 31),
            (39, 32),
            (39, 33),
            (39, 34),
            (39, 35),
            (39, 36),
            (39, 37),
            (39, 38),
            (39, 39),
            (39, 41),
            (39, 42),
            (39, 43),
            (39, 44),
            (39, 45),
            (39, 46),
            (39, 47),
            (39, 48),
            (39, 49),
            (39, 50),
            (39, 51),
            (39, 52),
            (39, 53),
            (39, 54),
            (39, 55),
            (39, 56),
            (39, 57),
            (39, 58),
            (39, 59),
            (39, 60),
            (39, 61),
            (39, 62),
            (39, 63),
            (39, 64),
            (39, 65),
            (39, 66),
            (39, 67),
            (39, 68),
            (39, 69),
            (39, 70),
            (39, 71),
            (39, 72),
            (39, 73),
            (39, 74),
            (39, 75),
            (39, 76),
            (39, 77),
            (39, 78),
            (39, 79),
        }:
            return 74
        elif key in {
            (0, 40),
            (1, 40),
            (2, 17),
            (5, 40),
            (9, 40),
            (10, 40),
            (11, 17),
            (14, 40),
            (16, 40),
            (17, 40),
            (20, 40),
            (21, 40),
            (23, 17),
            (25, 40),
            (26, 40),
            (28, 40),
            (29, 40),
            (30, 17),
            (31, 40),
            (32, 40),
            (34, 40),
            (37, 40),
            (39, 40),
            (40, 0),
            (40, 6),
            (40, 7),
            (40, 8),
            (40, 9),
            (40, 10),
            (40, 11),
            (40, 12),
            (40, 14),
            (40, 15),
            (40, 16),
            (40, 18),
            (40, 19),
            (40, 20),
            (40, 21),
            (40, 22),
            (40, 23),
            (40, 24),
            (40, 25),
            (40, 26),
            (40, 28),
            (40, 29),
            (40, 30),
            (40, 31),
            (40, 32),
            (40, 34),
            (40, 35),
            (40, 36),
            (40, 37),
            (40, 38),
            (40, 39),
            (40, 41),
            (40, 42),
            (40, 43),
            (40, 44),
            (40, 45),
            (40, 46),
            (40, 47),
            (40, 48),
            (40, 49),
            (40, 50),
            (40, 52),
            (40, 53),
            (40, 54),
            (40, 55),
            (40, 56),
            (40, 57),
            (40, 58),
            (40, 59),
            (40, 60),
            (40, 61),
            (40, 62),
            (40, 63),
            (40, 64),
            (40, 65),
            (40, 66),
            (40, 67),
            (40, 69),
            (40, 71),
            (40, 72),
            (40, 74),
            (40, 75),
            (40, 76),
            (40, 77),
            (40, 78),
            (40, 79),
            (41, 40),
            (43, 40),
            (44, 40),
            (45, 40),
            (47, 40),
            (48, 40),
            (49, 40),
            (50, 40),
            (51, 40),
            (52, 40),
            (53, 40),
            (54, 40),
            (55, 40),
            (56, 40),
            (57, 40),
            (58, 40),
            (59, 40),
            (60, 40),
            (61, 40),
            (62, 40),
            (63, 40),
            (64, 40),
            (65, 40),
            (66, 40),
            (67, 40),
            (69, 40),
            (71, 40),
            (72, 40),
            (73, 40),
            (74, 17),
            (74, 27),
            (74, 39),
            (75, 40),
            (76, 40),
            (77, 40),
            (78, 40),
            (79, 40),
        }:
            return 56
        elif key in {
            (5, 0),
            (5, 1),
            (5, 8),
            (5, 13),
            (5, 15),
            (5, 16),
            (5, 19),
            (5, 29),
            (5, 34),
            (5, 35),
            (5, 36),
            (5, 38),
            (5, 39),
            (5, 42),
            (5, 45),
            (5, 51),
            (5, 54),
            (5, 55),
            (5, 57),
            (5, 58),
            (5, 64),
            (5, 72),
            (5, 79),
            (54, 33),
        }:
            return 37
        elif key in {
            (5, 9),
            (5, 21),
            (5, 28),
            (5, 37),
            (5, 41),
            (5, 43),
            (5, 44),
            (5, 47),
            (5, 52),
            (5, 59),
            (5, 60),
            (5, 65),
            (5, 66),
            (5, 67),
            (45, 33),
            (50, 33),
            (59, 33),
        }:
            return 41
        elif key in {
            (2, 40),
            (8, 40),
            (11, 40),
            (12, 40),
            (23, 40),
            (30, 40),
            (40, 17),
            (40, 27),
            (40, 40),
            (42, 40),
            (46, 40),
            (74, 40),
        }:
            return 5
        elif key in {(5, 26), (5, 33), (5, 50)}:
            return 53
        return 24

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_1_outputs, attn_2_2_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_0_output, attn_2_3_output):
        key = (attn_2_0_output, attn_2_3_output)
        if key in {
            ("(", 0),
            ("(", 1),
            ("(", 4),
            ("(", 6),
            ("(", 10),
            ("(", 11),
            ("(", 12),
            ("(", 13),
            ("(", 22),
            ("(", 24),
            ("(", 31),
            ("(", 40),
            ("(", 44),
            ("(", 52),
            ("(", 60),
            ("(", 65),
            ("(", 73),
            ("(", 76),
            ("(", 79),
            (")", 1),
            (")", 4),
            ("<s>", 0),
            ("<s>", 1),
            ("<s>", 2),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 6),
            ("<s>", 7),
            ("<s>", 8),
            ("<s>", 9),
            ("<s>", 10),
            ("<s>", 11),
            ("<s>", 12),
            ("<s>", 13),
            ("<s>", 14),
            ("<s>", 15),
            ("<s>", 16),
            ("<s>", 17),
            ("<s>", 18),
            ("<s>", 19),
            ("<s>", 20),
            ("<s>", 21),
            ("<s>", 22),
            ("<s>", 23),
            ("<s>", 24),
            ("<s>", 25),
            ("<s>", 26),
            ("<s>", 28),
            ("<s>", 29),
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
            ("<s>", 50),
            ("<s>", 51),
            ("<s>", 52),
            ("<s>", 53),
            ("<s>", 54),
            ("<s>", 55),
            ("<s>", 56),
            ("<s>", 57),
            ("<s>", 58),
            ("<s>", 59),
            ("<s>", 60),
            ("<s>", 61),
            ("<s>", 62),
            ("<s>", 63),
            ("<s>", 64),
            ("<s>", 65),
            ("<s>", 66),
            ("<s>", 67),
            ("<s>", 68),
            ("<s>", 69),
            ("<s>", 70),
            ("<s>", 71),
            ("<s>", 72),
            ("<s>", 73),
            ("<s>", 74),
            ("<s>", 75),
            ("<s>", 76),
            ("<s>", 77),
            ("<s>", 78),
            ("<s>", 79),
            ("{", 0),
            ("{", 1),
            ("{", 3),
            ("{", 4),
            ("{", 6),
            ("{", 7),
            ("{", 8),
            ("{", 9),
            ("{", 10),
            ("{", 11),
            ("{", 12),
            ("{", 13),
            ("{", 14),
            ("{", 15),
            ("{", 16),
            ("{", 17),
            ("{", 18),
            ("{", 19),
            ("{", 20),
            ("{", 21),
            ("{", 22),
            ("{", 23),
            ("{", 24),
            ("{", 25),
            ("{", 26),
            ("{", 28),
            ("{", 29),
            ("{", 30),
            ("{", 31),
            ("{", 32),
            ("{", 34),
            ("{", 35),
            ("{", 36),
            ("{", 37),
            ("{", 38),
            ("{", 39),
            ("{", 40),
            ("{", 41),
            ("{", 42),
            ("{", 43),
            ("{", 44),
            ("{", 45),
            ("{", 46),
            ("{", 47),
            ("{", 48),
            ("{", 49),
            ("{", 50),
            ("{", 51),
            ("{", 52),
            ("{", 53),
            ("{", 54),
            ("{", 55),
            ("{", 56),
            ("{", 57),
            ("{", 58),
            ("{", 59),
            ("{", 60),
            ("{", 61),
            ("{", 62),
            ("{", 64),
            ("{", 65),
            ("{", 66),
            ("{", 68),
            ("{", 69),
            ("{", 70),
            ("{", 71),
            ("{", 72),
            ("{", 73),
            ("{", 74),
            ("{", 76),
            ("{", 77),
            ("{", 78),
            ("{", 79),
            ("}", 4),
        }:
            return 9
        return 55

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_0_outputs, attn_2_3_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(attn_2_0_output, position):
        key = (attn_2_0_output, position)
        if key in {
            ("(", 1),
            ("(", 3),
            ("(", 5),
            ("(", 12),
            ("(", 13),
            ("(", 29),
            (")", 3),
            (")", 5),
            (")", 12),
            (")", 13),
            (")", 29),
            ("<s>", 29),
            ("{", 29),
            ("}", 0),
            ("}", 1),
            ("}", 3),
            ("}", 5),
            ("}", 12),
            ("}", 13),
            ("}", 23),
            ("}", 29),
            ("}", 41),
            ("}", 42),
            ("}", 43),
            ("}", 44),
            ("}", 45),
            ("}", 46),
            ("}", 49),
            ("}", 50),
            ("}", 51),
            ("}", 52),
            ("}", 54),
            ("}", 55),
            ("}", 56),
            ("}", 58),
            ("}", 59),
            ("}", 60),
            ("}", 61),
            ("}", 63),
            ("}", 65),
            ("}", 67),
            ("}", 68),
            ("}", 69),
            ("}", 70),
            ("}", 71),
            ("}", 72),
            ("}", 74),
            ("}", 75),
            ("}", 77),
        }:
            return 6
        elif key in {
            ("(", 39),
            (")", 7),
            (")", 11),
            (")", 14),
            (")", 15),
            (")", 39),
            ("<s>", 39),
            ("{", 39),
            ("}", 7),
            ("}", 11),
            ("}", 14),
            ("}", 15),
            ("}", 39),
        }:
            return 19
        elif key in {(")", 2), (")", 9), (")", 10), ("}", 2), ("}", 9), ("}", 10)}:
            return 1
        elif key in {(")", 1), ("}", 21)}:
            return 16
        elif key in {
            ("}", 40),
            ("}", 47),
            ("}", 48),
            ("}", 53),
            ("}", 57),
            ("}", 62),
            ("}", 64),
            ("}", 66),
            ("}", 73),
            ("}", 76),
            ("}", 78),
            ("}", 79),
        }:
            return 37
        return 50

    mlp_2_2_outputs = [mlp_2_2(k0, k1) for k0, k1 in zip(attn_2_0_outputs, positions)]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(token, attn_2_3_output):
        key = (token, attn_2_3_output)
        return 44

    mlp_2_3_outputs = [mlp_2_3(k0, k1) for k0, k1 in zip(tokens, attn_2_3_outputs)]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_3_output, num_attn_2_0_output):
        key = (num_attn_2_3_output, num_attn_2_0_output)
        return 27

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_2_0_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_2_output, num_attn_2_0_output):
        key = (num_attn_1_2_output, num_attn_2_0_output)
        return 24

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_2_0_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_2_2_output):
        key = num_attn_2_2_output
        return 3

    num_mlp_2_2_outputs = [num_mlp_2_2(k0) for k0 in num_attn_2_2_outputs]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_2_3_output, num_attn_2_0_output):
        key = (num_attn_2_3_output, num_attn_2_0_output)
        return 23

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_2_0_outputs)
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
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
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
            "}",
            "(",
            "(",
            "(",
            ")",
            ")",
            "}",
            "{",
            "}",
            ")",
            ")",
            "{",
            ")",
            "}",
            "{",
            "(",
            "(",
            "(",
            ")",
            "}",
            "{",
            "(",
            "{",
            ")",
            "{",
            "{",
            "{",
            "{",
            "}",
            "(",
            ")",
            "}",
            "{",
            ")",
            "(",
            "{",
            ")",
        ]
    )
)
