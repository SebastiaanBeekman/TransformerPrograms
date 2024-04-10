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
        "output/length/rasp/dyck2/trainlength30/s4/dyck2_weights.csv",
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
        if q_position in {0, 36, 9, 11, 12, 53}:
            return k_position == 5
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 34, 4, 42, 58, 59}:
            return k_position == 2
        elif q_position in {35, 3, 5, 38, 7, 40, 41, 43, 48, 51, 52, 54, 57, 30}:
            return k_position == 3
        elif q_position in {8, 6}:
            return k_position == 4
        elif q_position in {10, 13, 16, 18, 31}:
            return k_position == 6
        elif q_position in {17, 14, 15}:
            return k_position == 7
        elif q_position in {27, 26, 19, 28}:
            return k_position == 9
        elif q_position in {24, 25, 20, 22}:
            return k_position == 8
        elif q_position in {21, 23}:
            return k_position == 10
        elif q_position in {29}:
            return k_position == 12
        elif q_position in {32}:
            return k_position == 51
        elif q_position in {33}:
            return k_position == 57
        elif q_position in {37}:
            return k_position == 45
        elif q_position in {39}:
            return k_position == 39
        elif q_position in {56, 44}:
            return k_position == 47
        elif q_position in {45}:
            return k_position == 38
        elif q_position in {46, 55}:
            return k_position == 49
        elif q_position in {47}:
            return k_position == 41
        elif q_position in {49}:
            return k_position == 40
        elif q_position in {50}:
            return k_position == 33

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_token, k_token):
        if q_token in {")", "("}:
            return k_token == ")"
        elif q_token in {"<s>"}:
            return k_token == "("
        elif q_token in {"}", "{"}:
            return k_token == "}"

    attn_0_1_pattern = select_closest(tokens, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 58, 47}:
            return k_position == 36
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {59, 3, 7}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {9, 5}:
            return k_position == 4
        elif q_position in {8, 10, 6}:
            return k_position == 5
        elif q_position in {17, 11, 12, 14}:
            return k_position == 6
        elif q_position in {16, 18, 20, 13}:
            return k_position == 7
        elif q_position in {29, 15}:
            return k_position == 8
        elif q_position in {26, 19, 28}:
            return k_position == 10
        elif q_position in {24, 27, 21, 22}:
            return k_position == 9
        elif q_position in {25, 23}:
            return k_position == 11
        elif q_position in {30, 31}:
            return k_position == 57
        elif q_position in {32}:
            return k_position == 30
        elif q_position in {33, 43}:
            return k_position == 56
        elif q_position in {34, 38}:
            return k_position == 29
        elif q_position in {49, 35}:
            return k_position == 23
        elif q_position in {36}:
            return k_position == 58
        elif q_position in {37}:
            return k_position == 59
        elif q_position in {39}:
            return k_position == 34
        elif q_position in {40}:
            return k_position == 44
        elif q_position in {41}:
            return k_position == 50
        elif q_position in {42}:
            return k_position == 32
        elif q_position in {44}:
            return k_position == 37
        elif q_position in {45}:
            return k_position == 55
        elif q_position in {46}:
            return k_position == 43
        elif q_position in {48}:
            return k_position == 40
        elif q_position in {50}:
            return k_position == 26
        elif q_position in {51}:
            return k_position == 42
        elif q_position in {52}:
            return k_position == 45
        elif q_position in {53}:
            return k_position == 51
        elif q_position in {54}:
            return k_position == 31
        elif q_position in {55}:
            return k_position == 48
        elif q_position in {56}:
            return k_position == 54
        elif q_position in {57}:
            return k_position == 41

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {
            0,
            32,
            35,
            6,
            38,
            39,
            40,
            43,
            44,
            45,
            47,
            48,
            50,
            51,
            54,
            55,
            56,
            31,
        }:
            return k_position == 5
        elif q_position in {8, 1}:
            return k_position == 7
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {33, 34, 36, 37, 7, 41, 42, 46, 49, 52, 53, 58, 59, 30}:
            return k_position == 6
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 9
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
        elif q_position in {57}:
            return k_position == 31

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"("}:
            return position == 46
        elif token in {")", "}"}:
            return position == 9
        elif token in {"<s>"}:
            return position == 57
        elif token in {"{"}:
            return position == 58

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {
            0,
            1,
            2,
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
            19,
            20,
            21,
            22,
            23,
            24,
            26,
            27,
            28,
            29,
            37,
            42,
            48,
            53,
            54,
        }:
            return token == ""
        elif position in {33, 3, 35, 43, 50, 52, 58, 59, 30}:
            return token == "{"
        elif position in {
            32,
            34,
            36,
            6,
            38,
            39,
            40,
            41,
            44,
            45,
            46,
            47,
            49,
            51,
            55,
            56,
            57,
            31,
        }:
            return token == "("
        elif position in {25}:
            return token == "}"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"("}:
            return position == 48
        elif token in {")", "}"}:
            return position == 10
        elif token in {"<s>"}:
            return position == 28
        elif token in {"{"}:
            return position == 29

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"("}:
            return position == 13
        elif token in {")"}:
            return position == 26
        elif token in {"<s>"}:
            return position == 2
        elif token in {"{"}:
            return position == 12
        elif token in {"}"}:
            return position == 25

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(token, position):
        key = (token, position)
        if key in {
            ("(", 0),
            ("(", 3),
            ("(", 5),
            ("(", 6),
            ("(", 7),
            ("(", 8),
            ("(", 9),
            ("(", 10),
            ("(", 11),
            ("(", 12),
            ("(", 13),
            ("(", 14),
            ("(", 15),
            ("(", 16),
            ("(", 17),
            ("(", 18),
            ("(", 20),
            ("(", 21),
            ("(", 22),
            ("(", 23),
            ("(", 24),
            ("(", 25),
            ("(", 26),
            ("(", 27),
            ("(", 28),
            ("(", 29),
            ("(", 30),
            ("(", 31),
            ("(", 32),
            ("(", 33),
            ("(", 34),
            ("(", 35),
            ("(", 36),
            ("(", 37),
            ("(", 38),
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
            (")", 9),
            (")", 15),
            (")", 17),
            (")", 21),
            (")", 27),
            (")", 29),
            ("<s>", 7),
            ("<s>", 9),
            ("<s>", 15),
            ("<s>", 17),
            ("<s>", 21),
            ("<s>", 25),
            ("<s>", 27),
            ("<s>", 29),
            ("{", 0),
            ("{", 3),
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
            ("{", 20),
            ("{", 21),
            ("{", 22),
            ("{", 23),
            ("{", 24),
            ("{", 25),
            ("{", 26),
            ("{", 27),
            ("{", 28),
            ("{", 29),
            ("{", 30),
            ("{", 31),
            ("{", 32),
            ("{", 33),
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
            ("}", 7),
            ("}", 9),
            ("}", 15),
            ("}", 17),
            ("}", 21),
            ("}", 25),
            ("}", 27),
            ("}", 29),
        }:
            return 15
        elif key in {
            ("(", 19),
            (")", 7),
            (")", 19),
            (")", 23),
            (")", 25),
            ("<s>", 19),
            ("<s>", 23),
            ("}", 19),
            ("}", 23),
        }:
            return 26
        elif key in {
            ("(", 1),
            ("(", 2),
            ("(", 4),
            ("{", 1),
            ("{", 2),
            ("{", 4),
            ("{", 19),
        }:
            return 10
        return 43

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(tokens, positions)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(token, position):
        key = (token, position)
        if key in {
            ("(", 15),
            (")", 1),
            (")", 2),
            (")", 5),
            (")", 6),
            (")", 7),
            (")", 8),
            (")", 9),
            (")", 10),
            (")", 12),
            (")", 14),
            (")", 15),
            (")", 18),
            (")", 20),
            (")", 24),
            (")", 27),
            (")", 28),
            (")", 29),
            ("<s>", 6),
            ("<s>", 8),
            ("<s>", 9),
            ("<s>", 15),
            ("{", 0),
            ("{", 1),
            ("{", 2),
            ("{", 3),
            ("{", 4),
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
            ("{", 27),
            ("{", 28),
            ("{", 29),
            ("{", 30),
            ("{", 31),
            ("{", 32),
            ("{", 33),
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
            ("}", 0),
            ("}", 1),
            ("}", 2),
            ("}", 3),
            ("}", 4),
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
            ("}", 27),
            ("}", 28),
            ("}", 29),
            ("}", 30),
            ("}", 31),
            ("}", 32),
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
        }:
            return 4
        return 24

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(tokens, positions)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_0_output):
        key = attn_0_0_output
        return 46

    mlp_0_2_outputs = [mlp_0_2(k0) for k0 in attn_0_0_outputs]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_2_output, attn_0_0_output):
        key = (attn_0_2_output, attn_0_0_output)
        if key in {("(", "("), (")", "("), ("<s>", "(")}:
            return 9
        elif key in {
            ("(", ")"),
            ("(", "}"),
            (")", ")"),
            (")", "}"),
            ("<s>", ")"),
            ("<s>", "}"),
            ("{", ")"),
            ("{", "}"),
            ("}", ")"),
            ("}", "}"),
        }:
            return 2
        elif key in {("}", "("), ("}", "<s>")}:
            return 5
        return 22

    mlp_0_3_outputs = [
        mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_0_outputs)
    ]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output, num_attn_0_3_output):
        key = (num_attn_0_0_output, num_attn_0_3_output)
        return 9

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_3_output, num_attn_0_2_output):
        key = (num_attn_0_3_output, num_attn_0_2_output)
        return 43

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_3_output, num_attn_0_2_output):
        key = (num_attn_0_3_output, num_attn_0_2_output)
        return 49

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_1_output):
        key = num_attn_0_1_output
        return 18

    num_mlp_0_3_outputs = [num_mlp_0_3(k0) for k0 in num_attn_0_1_outputs]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(token, position):
        if token in {"("}:
            return position == 3
        elif token in {")"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 5
        elif token in {"{"}:
            return position == 4
        elif token in {"}"}:
            return position == 7

    attn_1_0_pattern = select_closest(positions, tokens, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_0_output, position):
        if attn_0_0_output in {"("}:
            return position == 5
        elif attn_0_0_output in {")"}:
            return position == 2
        elif attn_0_0_output in {"<s>"}:
            return position == 4
        elif attn_0_0_output in {"{"}:
            return position == 7
        elif attn_0_0_output in {"}"}:
            return position == 1

    attn_1_1_pattern = select_closest(positions, attn_0_0_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_0_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(token, position):
        if token in {"{", "("}:
            return position == 1
        elif token in {")", "<s>"}:
            return position == 4
        elif token in {"}"}:
            return position == 3

    attn_1_2_pattern = select_closest(positions, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_0_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_3_output, position):
        if attn_0_3_output in {"("}:
            return position == 8
        elif attn_0_3_output in {")", "}"}:
            return position == 5
        elif attn_0_3_output in {"<s>", "{"}:
            return position == 1

    attn_1_3_pattern = select_closest(positions, attn_0_3_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(mlp_0_0_output, attn_0_3_output):
        if mlp_0_0_output in {
            0,
            1,
            3,
            4,
            5,
            6,
            7,
            9,
            10,
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
            26,
            27,
            29,
            30,
            31,
            33,
            34,
            36,
            37,
            39,
            40,
            41,
            42,
            44,
            48,
            49,
            50,
            51,
            52,
            53,
            55,
            56,
            57,
            58,
        }:
            return attn_0_3_output == ""
        elif mlp_0_0_output in {2, 35, 38, 43, 47, 28}:
            return attn_0_3_output == "}"
        elif mlp_0_0_output in {32, 8, 11, 45, 46, 54, 24, 59}:
            return attn_0_3_output == ")"

    num_attn_1_0_pattern = select(attn_0_3_outputs, mlp_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_0_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_0_output, attn_0_3_output):
        if attn_0_0_output in {"{", "("}:
            return attn_0_3_output == ""
        elif attn_0_0_output in {")"}:
            return attn_0_3_output == "("
        elif attn_0_0_output in {"<s>", "}"}:
            return attn_0_3_output == "{"

    num_attn_1_1_pattern = select(attn_0_3_outputs, attn_0_0_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_0_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_2_output, attn_0_3_output):
        if attn_0_2_output in {"{", "<s>", "("}:
            return attn_0_3_output == ""
        elif attn_0_2_output in {")"}:
            return attn_0_3_output == "("
        elif attn_0_2_output in {"}"}:
            return attn_0_3_output == "{"

    num_attn_1_2_pattern = select(attn_0_3_outputs, attn_0_2_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(q_attn_0_3_output, k_attn_0_3_output):
        if q_attn_0_3_output in {"{", "("}:
            return k_attn_0_3_output == ""
        elif q_attn_0_3_output in {")"}:
            return k_attn_0_3_output == "("
        elif q_attn_0_3_output in {"<s>"}:
            return k_attn_0_3_output == "<s>"
        elif q_attn_0_3_output in {"}"}:
            return k_attn_0_3_output == "{"

    num_attn_1_3_pattern = select(attn_0_3_outputs, attn_0_3_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_0_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_2_output, attn_1_3_output):
        key = (attn_0_2_output, attn_1_3_output)
        if key in {("<s>", "("), ("<s>", "<s>"), ("<s>", "{"), ("<s>", "}")}:
            return 24
        return 0

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_1_3_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_1_output, attn_1_3_output):
        key = (attn_1_1_output, attn_1_3_output)
        if key in {
            ("(", "("),
            ("(", ")"),
            ("(", "<s>"),
            ("(", "{"),
            ("(", "}"),
            (")", "("),
            (")", ")"),
            (")", "<s>"),
            (")", "{"),
            (")", "}"),
            ("<s>", "("),
            ("<s>", ")"),
            ("<s>", "<s>"),
            ("<s>", "{"),
            ("<s>", "}"),
            ("{", "("),
            ("{", ")"),
            ("{", "<s>"),
            ("{", "{"),
            ("{", "}"),
            ("}", "("),
            ("}", ")"),
            ("}", "<s>"),
            ("}", "{"),
            ("}", "}"),
        }:
            return 8
        return 7

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_1_3_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(token, attn_0_3_output):
        key = (token, attn_0_3_output)
        if key in {
            ("(", "{"),
            (")", ")"),
            (")", "<s>"),
            (")", "{"),
            (")", "}"),
            ("<s>", ")"),
            ("<s>", "{"),
            ("<s>", "}"),
            ("{", "{"),
        }:
            return 36
        return 31

    mlp_1_2_outputs = [mlp_1_2(k0, k1) for k0, k1 in zip(tokens, attn_0_3_outputs)]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_1_2_output, attn_1_1_output):
        key = (attn_1_2_output, attn_1_1_output)
        if key in {("(", "("), ("(", "<s>"), (")", "("), ("<s>", "("), ("<s>", "<s>")}:
            return 27
        elif key in {("(", ")"), ("{", "{"), ("{", "}"), ("}", "{")}:
            return 42
        elif key in {("(", "}"), ("<s>", "}"), ("}", "("), ("}", "<s>"), ("}", "}")}:
            return 25
        elif key in {(")", ")"), (")", "<s>"), (")", "{"), (")", "}"), ("<s>", ")")}:
            return 52
        elif key in {("{", ")"), ("}", ")")}:
            return 17
        return 8

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_1_1_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_0_output, num_attn_1_2_output):
        key = (num_attn_1_0_output, num_attn_1_2_output)
        return 0

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_3_output, num_attn_0_1_output):
        key = (num_attn_0_3_output, num_attn_0_1_output)
        return 52

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_0_1_output, num_attn_0_0_output):
        key = (num_attn_0_1_output, num_attn_0_0_output)
        return 59

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_0_output, num_attn_1_2_output):
        key = (num_attn_1_0_output, num_attn_1_2_output)
        return 16

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_position, k_position):
        if q_position in {
            0,
            15,
            17,
            18,
            20,
            22,
            27,
            29,
            30,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            40,
            42,
            43,
            45,
            47,
            51,
            52,
            53,
            55,
            56,
        }:
            return k_position == 6
        elif q_position in {1, 4, 7, 54, 58, 59, 31}:
            return k_position == 1
        elif q_position in {2, 5}:
            return k_position == 9
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {6}:
            return k_position == 3
        elif q_position in {8, 23}:
            return k_position == 4
        elif q_position in {39, 9, 41, 44, 48, 50, 24, 57}:
            return k_position == 7
        elif q_position in {10, 11, 12, 13, 14, 16, 19, 21, 25}:
            return k_position == 5
        elif q_position in {26}:
            return k_position == 16
        elif q_position in {28}:
            return k_position == 14
        elif q_position in {46}:
            return k_position == 8
        elif q_position in {49}:
            return k_position == 35

    attn_2_0_pattern = select_closest(positions, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(token, position):
        if token in {"{", "("}:
            return position == 1
        elif token in {")"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 6
        elif token in {"}"}:
            return position == 2

    attn_2_1_pattern = select_closest(positions, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_2_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_position, k_position):
        if q_position in {
            0,
            9,
            10,
            12,
            15,
            19,
            21,
            22,
            29,
            31,
            34,
            35,
            42,
            44,
            46,
            48,
            49,
            51,
            52,
            56,
        }:
            return k_position == 7
        elif q_position in {1, 4, 6, 7}:
            return k_position == 3
        elif q_position in {25, 2, 26}:
            return k_position == 9
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 10
        elif q_position in {
            8,
            11,
            13,
            14,
            17,
            18,
            20,
            23,
            27,
            28,
            30,
            32,
            33,
            36,
            37,
            38,
            39,
            40,
            41,
            43,
            45,
            47,
            50,
            53,
            54,
            55,
            57,
            58,
            59,
        }:
            return k_position == 6
        elif q_position in {16}:
            return k_position == 5
        elif q_position in {24}:
            return k_position == 8

    attn_2_2_pattern = select_closest(positions, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_2_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_3_output, position):
        if attn_0_3_output in {"("}:
            return position == 1
        elif attn_0_3_output in {")", "}"}:
            return position == 2
        elif attn_0_3_output in {"<s>"}:
            return position == 7
        elif attn_0_3_output in {"{"}:
            return position == 9

    attn_2_3_pattern = select_closest(positions, attn_0_3_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_1_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(num_mlp_1_1_output, attn_0_0_output):
        if num_mlp_1_1_output in {
            0,
            2,
            4,
            8,
            12,
            14,
            20,
            23,
            28,
            29,
            31,
            33,
            34,
            42,
            44,
            45,
            48,
            54,
            56,
            58,
            59,
        }:
            return attn_0_0_output == "}"
        elif num_mlp_1_1_output in {
            1,
            3,
            5,
            6,
            7,
            9,
            10,
            11,
            13,
            15,
            16,
            17,
            18,
            19,
            21,
            22,
            24,
            26,
            27,
            30,
            32,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            43,
            46,
            47,
            49,
            50,
            51,
            52,
            53,
            55,
            57,
        }:
            return attn_0_0_output == ")"
        elif num_mlp_1_1_output in {25}:
            return attn_0_0_output == "<s>"

    num_attn_2_0_pattern = select(
        attn_0_0_outputs, num_mlp_1_1_outputs, num_predicate_2_0
    )
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_0_1_output, attn_1_2_output):
        if attn_0_1_output in {"("}:
            return attn_1_2_output == "<s>"
        elif attn_0_1_output in {")"}:
            return attn_1_2_output == "("
        elif attn_0_1_output in {"<s>", "{"}:
            return attn_1_2_output == ""
        elif attn_0_1_output in {"}"}:
            return attn_1_2_output == "{"

    num_attn_2_1_pattern = select(attn_1_2_outputs, attn_0_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_2_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(mlp_1_0_output, attn_1_0_output):
        if mlp_1_0_output in {
            0,
            2,
            4,
            6,
            7,
            8,
            9,
            10,
            13,
            15,
            16,
            17,
            18,
            21,
            22,
            23,
            24,
            25,
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
            39,
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
            54,
            56,
            57,
            58,
            59,
        }:
            return attn_1_0_output == "<s>"
        elif mlp_1_0_output in {1, 5, 38, 41, 14, 50, 19, 53, 55, 26, 31}:
            return attn_1_0_output == "{"
        elif mlp_1_0_output in {3, 40, 11, 12, 20}:
            return attn_1_0_output == "("

    num_attn_2_2_pattern = select(attn_1_0_outputs, mlp_1_0_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_3_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(mlp_1_1_output, attn_1_0_output):
        if mlp_1_1_output in {
            0,
            8,
            9,
            10,
            11,
            12,
            17,
            19,
            20,
            23,
            29,
            30,
            32,
            33,
            34,
            38,
            39,
            41,
            43,
            44,
            46,
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
            return attn_1_0_output == "<s>"
        elif mlp_1_1_output in {1, 2, 5, 37, 7, 42, 47, 49, 54, 26}:
            return attn_1_0_output == ""
        elif mlp_1_1_output in {3, 4, 35, 13, 14, 15, 45, 18, 21, 24, 28}:
            return attn_1_0_output == "("
        elif mlp_1_1_output in {6, 40, 16, 51, 22, 25, 27, 31}:
            return attn_1_0_output == "{"
        elif mlp_1_1_output in {36}:
            return attn_1_0_output == "<pad>"

    num_attn_2_3_pattern = select(attn_1_0_outputs, mlp_1_1_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_2_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_2_output, attn_2_0_output):
        key = (attn_2_2_output, attn_2_0_output)
        if key in {
            ("(", ")"),
            ("(", "}"),
            (")", "("),
            (")", ")"),
            (")", "<s>"),
            (")", "{"),
            (")", "}"),
            ("<s>", "("),
            ("<s>", ")"),
            ("<s>", "<s>"),
            ("<s>", "{"),
            ("<s>", "}"),
            ("{", ")"),
            ("{", "}"),
            ("}", ")"),
        }:
            return 51
        elif key in {("}", "("), ("}", "<s>"), ("}", "{"), ("}", "}")}:
            return 4
        return 5

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_2_outputs, attn_2_0_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(num_mlp_1_3_output, mlp_0_2_output):
        key = (num_mlp_1_3_output, mlp_0_2_output)
        if key in {
            (0, 31),
            (1, 31),
            (2, 31),
            (3, 8),
            (3, 9),
            (3, 10),
            (3, 11),
            (3, 13),
            (3, 19),
            (3, 27),
            (3, 28),
            (3, 30),
            (3, 31),
            (3, 41),
            (3, 44),
            (3, 51),
            (3, 54),
            (3, 55),
            (3, 56),
            (3, 58),
            (4, 31),
            (4, 51),
            (5, 30),
            (5, 31),
            (5, 51),
            (6, 31),
            (7, 10),
            (7, 11),
            (7, 13),
            (7, 19),
            (7, 28),
            (7, 30),
            (7, 31),
            (7, 41),
            (7, 51),
            (8, 31),
            (9, 30),
            (9, 31),
            (9, 51),
            (10, 8),
            (10, 9),
            (10, 10),
            (10, 11),
            (10, 12),
            (10, 13),
            (10, 19),
            (10, 27),
            (10, 28),
            (10, 30),
            (10, 31),
            (10, 41),
            (10, 51),
            (10, 54),
            (10, 55),
            (10, 58),
            (11, 31),
            (12, 31),
            (13, 31),
            (14, 31),
            (15, 30),
            (15, 31),
            (15, 51),
            (17, 2),
            (17, 8),
            (17, 9),
            (17, 10),
            (17, 11),
            (17, 13),
            (17, 19),
            (17, 20),
            (17, 27),
            (17, 28),
            (17, 29),
            (17, 30),
            (17, 31),
            (17, 35),
            (17, 41),
            (17, 44),
            (17, 51),
            (17, 52),
            (17, 54),
            (17, 55),
            (17, 56),
            (17, 58),
            (18, 10),
            (18, 11),
            (18, 13),
            (18, 19),
            (18, 28),
            (18, 30),
            (18, 31),
            (18, 51),
            (19, 31),
            (20, 31),
            (21, 0),
            (21, 2),
            (21, 8),
            (21, 9),
            (21, 10),
            (21, 11),
            (21, 12),
            (21, 13),
            (21, 15),
            (21, 19),
            (21, 20),
            (21, 27),
            (21, 28),
            (21, 29),
            (21, 30),
            (21, 31),
            (21, 35),
            (21, 38),
            (21, 41),
            (21, 43),
            (21, 44),
            (21, 51),
            (21, 52),
            (21, 54),
            (21, 55),
            (21, 56),
            (21, 58),
            (22, 1),
            (22, 2),
            (22, 3),
            (22, 4),
            (22, 5),
            (22, 6),
            (22, 8),
            (22, 9),
            (22, 10),
            (22, 11),
            (22, 13),
            (22, 14),
            (22, 16),
            (22, 17),
            (22, 18),
            (22, 21),
            (22, 22),
            (22, 23),
            (22, 24),
            (22, 25),
            (22, 26),
            (22, 27),
            (22, 30),
            (22, 31),
            (22, 32),
            (22, 34),
            (22, 35),
            (22, 36),
            (22, 37),
            (22, 40),
            (22, 41),
            (22, 42),
            (22, 44),
            (22, 45),
            (22, 47),
            (22, 55),
            (22, 57),
            (22, 58),
            (23, 31),
            (24, 31),
            (24, 51),
            (25, 31),
            (26, 11),
            (26, 13),
            (26, 19),
            (26, 27),
            (26, 28),
            (26, 30),
            (26, 31),
            (26, 41),
            (26, 51),
            (26, 58),
            (27, 31),
            (27, 51),
            (28, 31),
            (29, 4),
            (29, 14),
            (29, 22),
            (29, 31),
            (29, 32),
            (29, 36),
            (29, 44),
            (29, 58),
            (30, 31),
            (31, 31),
            (32, 4),
            (32, 14),
            (32, 22),
            (32, 31),
            (32, 32),
            (32, 36),
            (32, 58),
            (33, 31),
            (33, 51),
            (34, 31),
            (35, 31),
            (36, 31),
            (37, 31),
            (38, 11),
            (38, 19),
            (38, 30),
            (38, 31),
            (38, 51),
            (39, 11),
            (39, 19),
            (39, 30),
            (39, 31),
            (39, 51),
            (40, 31),
            (41, 10),
            (41, 11),
            (41, 13),
            (41, 19),
            (41, 28),
            (41, 30),
            (41, 31),
            (41, 51),
            (42, 31),
            (43, 31),
            (44, 31),
            (44, 51),
            (45, 31),
            (46, 30),
            (46, 31),
            (46, 51),
            (47, 31),
            (48, 31),
            (48, 51),
            (49, 11),
            (49, 19),
            (49, 30),
            (49, 31),
            (49, 51),
            (50, 31),
            (51, 10),
            (51, 11),
            (51, 19),
            (51, 30),
            (51, 31),
            (51, 51),
            (52, 10),
            (52, 30),
            (52, 31),
            (52, 51),
            (53, 31),
            (54, 31),
            (55, 31),
            (56, 30),
            (56, 31),
            (56, 51),
            (57, 31),
            (58, 11),
            (58, 13),
            (58, 19),
            (58, 27),
            (58, 28),
            (58, 30),
            (58, 31),
            (58, 41),
            (58, 51),
            (58, 54),
            (58, 55),
            (58, 58),
            (59, 31),
        }:
            return 18
        elif key in {
            (0, 53),
            (1, 53),
            (2, 33),
            (2, 53),
            (4, 53),
            (5, 53),
            (7, 53),
            (10, 14),
            (10, 33),
            (10, 45),
            (10, 53),
            (11, 53),
            (12, 0),
            (12, 14),
            (12, 33),
            (12, 42),
            (12, 45),
            (12, 53),
            (12, 58),
            (14, 14),
            (14, 33),
            (14, 45),
            (14, 53),
            (15, 53),
            (16, 53),
            (17, 53),
            (18, 53),
            (19, 53),
            (20, 53),
            (21, 53),
            (22, 7),
            (22, 12),
            (22, 33),
            (22, 53),
            (22, 54),
            (22, 56),
            (22, 59),
            (23, 53),
            (25, 53),
            (27, 53),
            (28, 53),
            (29, 13),
            (29, 18),
            (29, 30),
            (29, 35),
            (32, 13),
            (32, 18),
            (32, 30),
            (32, 35),
            (32, 44),
            (32, 53),
            (33, 53),
            (34, 53),
            (35, 53),
            (36, 53),
            (37, 33),
            (37, 53),
            (38, 53),
            (39, 53),
            (40, 0),
            (40, 3),
            (40, 8),
            (40, 9),
            (40, 13),
            (40, 14),
            (40, 15),
            (40, 23),
            (40, 27),
            (40, 28),
            (40, 32),
            (40, 33),
            (40, 42),
            (40, 45),
            (41, 14),
            (41, 33),
            (41, 53),
            (43, 53),
            (44, 53),
            (46, 53),
            (47, 14),
            (47, 33),
            (47, 42),
            (47, 45),
            (47, 53),
            (50, 33),
            (50, 53),
            (51, 53),
            (53, 53),
            (54, 53),
            (55, 14),
            (55, 33),
            (55, 42),
            (55, 45),
            (55, 53),
            (56, 53),
            (57, 53),
            (58, 53),
            (59, 53),
        }:
            return 19
        elif key in {(44, 5), (44, 30), (44, 32), (44, 33), (45, 30), (45, 32)}:
            return 11
        elif key in {(9, 53), (40, 53), (45, 53)}:
            return 17
        elif key in {(45, 33)}:
            return 16
        return 32

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(num_mlp_1_3_outputs, mlp_0_2_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(mlp_0_2_output, attn_0_2_output):
        key = (mlp_0_2_output, attn_0_2_output)
        if key in {(33, "("), (33, ")"), (33, "<s>"), (33, "{"), (33, "}")}:
            return 37
        return 55

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(mlp_0_2_outputs, attn_0_2_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(attn_1_1_output, attn_0_1_output):
        key = (attn_1_1_output, attn_0_1_output)
        if key in {("<s>", "{")}:
            return 51
        elif key in {("(", ")"), ("(", "{"), (")", ")"), (")", "{"), ("<s>", ")")}:
            return 38
        return 9

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_0_1_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_0_output, num_attn_1_2_output):
        key = (num_attn_1_0_output, num_attn_1_2_output)
        return 35

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_1_output, num_attn_1_3_output):
        key = (num_attn_2_1_output, num_attn_1_3_output)
        return 53

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_2_1_output, num_attn_1_0_output):
        key = (num_attn_2_1_output, num_attn_1_0_output)
        return 31

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_1_2_output, num_attn_1_1_output):
        key = (num_attn_1_2_output, num_attn_1_1_output)
        return 57

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_1_outputs)
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
            "{",
            "}",
            ")",
            ")",
            "(",
            "}",
            "(",
            "{",
            ")",
            "{",
            "(",
            "}",
            ")",
            "{",
            "}",
            "}",
            ")",
            "(",
            "(",
            "{",
            "(",
            "{",
            "{",
            "{",
            "(",
            "}",
            "(",
            "}",
            ")",
        ]
    )
)
