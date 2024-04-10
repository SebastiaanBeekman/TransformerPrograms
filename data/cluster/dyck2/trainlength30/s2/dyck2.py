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
        "output/length/rasp/dyck2/trainlength30/s2/dyck2_weights.csv",
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
        if q_position in {0, 41, 10, 12, 13, 14, 16, 18, 21, 30, 31}:
            return k_position == 5
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {32, 3, 4, 38, 7, 47, 58}:
            return k_position == 2
        elif q_position in {8, 5}:
            return k_position == 3
        elif q_position in {6, 9, 15, 17, 29}:
            return k_position == 4
        elif q_position in {11, 20, 22, 24, 27}:
            return k_position == 7
        elif q_position in {
            19,
            23,
            25,
            33,
            35,
            37,
            42,
            43,
            44,
            45,
            46,
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
            59,
        }:
            return k_position == 6
        elif q_position in {26, 28}:
            return k_position == 8
        elif q_position in {34}:
            return k_position == 41
        elif q_position in {36}:
            return k_position == 39
        elif q_position in {39}:
            return k_position == 43
        elif q_position in {40}:
            return k_position == 50

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 50
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 3, 7}:
            return k_position == 2
        elif q_position in {8, 4, 6}:
            return k_position == 3
        elif q_position in {10, 5}:
            return k_position == 0
        elif q_position in {9}:
            return k_position == 4
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
        elif q_position in {33, 35, 36, 40, 41, 43, 45, 49, 51, 52, 56, 30, 31}:
            return k_position == 5
        elif q_position in {32}:
            return k_position == 37
        elif q_position in {34, 37}:
            return k_position == 40
        elif q_position in {59, 38}:
            return k_position == 57
        elif q_position in {39}:
            return k_position == 56
        elif q_position in {42}:
            return k_position == 34
        elif q_position in {44}:
            return k_position == 33
        elif q_position in {46, 47}:
            return k_position == 30
        elif q_position in {48}:
            return k_position == 44
        elif q_position in {50, 53}:
            return k_position == 38
        elif q_position in {54}:
            return k_position == 51
        elif q_position in {55}:
            return k_position == 48
        elif q_position in {57}:
            return k_position == 45
        elif q_position in {58}:
            return k_position == 59

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_token, k_token):
        if q_token in {"{", "(", ")"}:
            return k_token == ")"
        elif q_token in {"<s>", "}"}:
            return k_token == "}"

    attn_0_2_pattern = select_closest(tokens, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 33, 34, 36, 41, 10, 11, 44, 50, 51, 55, 57, 58, 30, 31}:
            return k_position == 6
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 2
        elif q_position in {3, 4}:
            return k_position == 3
        elif q_position in {9, 5, 7}:
            return k_position == 4
        elif q_position in {32, 35, 6, 8, 45, 46, 47, 53, 54}:
            return k_position == 5
        elif q_position in {12, 14, 15, 16, 17, 18, 19, 21, 26, 29}:
            return k_position == 7
        elif q_position in {24, 20, 13, 22}:
            return k_position == 8
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {25}:
            return k_position == 24
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 9
        elif q_position in {37}:
            return k_position == 50
        elif q_position in {38}:
            return k_position == 47
        elif q_position in {42, 39}:
            return k_position == 53
        elif q_position in {40}:
            return k_position == 41
        elif q_position in {43}:
            return k_position == 43
        elif q_position in {48}:
            return k_position == 14
        elif q_position in {49}:
            return k_position == 42
        elif q_position in {52}:
            return k_position == 33
        elif q_position in {56}:
            return k_position == 45
        elif q_position in {59}:
            return k_position == 30

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"{", "("}:
            return position == 10
        elif token in {")"}:
            return position == 42
        elif token in {"<s>"}:
            return position == 43
        elif token in {"}"}:
            return position == 47

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
            13,
            14,
            17,
            18,
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
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            59,
        }:
            return token == ""
        elif position in {3, 4, 7, 10, 11, 12}:
            return token == "}"
        elif position in {5, 15, 16, 19, 20, 57}:
            return token == "<pad>"
        elif position in {48, 58, 6}:
            return token == "<s>"
        elif position in {8, 9, 35}:
            return token == ")"
        elif position in {32}:
            return token == "{"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"{", "("}:
            return position == 8
        elif token in {"}", ")"}:
            return position == 27
        elif token in {"<s>"}:
            return position == 19

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0, 48}:
            return k_position == 14
        elif q_position in {1, 2, 33, 5, 49, 56}:
            return k_position == 0
        elif q_position in {3, 13}:
            return k_position == 2
        elif q_position in {4, 44}:
            return k_position == 38
        elif q_position in {6}:
            return k_position == 22
        elif q_position in {45, 7}:
            return k_position == 40
        elif q_position in {8}:
            return k_position == 11
        elif q_position in {9}:
            return k_position == 26
        elif q_position in {10}:
            return k_position == 1
        elif q_position in {11, 28, 37}:
            return k_position == 51
        elif q_position in {12}:
            return k_position == 29
        elif q_position in {14}:
            return k_position == 28
        elif q_position in {15}:
            return k_position == 52
        elif q_position in {16}:
            return k_position == 23
        elif q_position in {17}:
            return k_position == 36
        elif q_position in {18, 23}:
            return k_position == 32
        elif q_position in {19}:
            return k_position == 37
        elif q_position in {20, 21}:
            return k_position == 56
        elif q_position in {22}:
            return k_position == 31
        elif q_position in {24}:
            return k_position == 50
        elif q_position in {25}:
            return k_position == 34
        elif q_position in {26}:
            return k_position == 49
        elif q_position in {27}:
            return k_position == 41
        elif q_position in {29, 47}:
            return k_position == 55
        elif q_position in {35, 30}:
            return k_position == 8
        elif q_position in {32, 36, 38, 39, 46, 50, 54, 59, 31}:
            return k_position == 6
        elif q_position in {34}:
            return k_position == 35
        elif q_position in {40, 41}:
            return k_position == 54
        elif q_position in {42, 43, 53}:
            return k_position == 7
        elif q_position in {51}:
            return k_position == 48
        elif q_position in {58, 52}:
            return k_position == 45
        elif q_position in {55}:
            return k_position == 16
        elif q_position in {57}:
            return k_position == 3

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_1_output, attn_0_2_output):
        key = (attn_0_1_output, attn_0_2_output)
        if key in {("{", "("), ("{", "<s>"), ("{", "{"), ("{", "}")}:
            return 48
        return 19

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_2_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_3_output, token):
        key = (attn_0_3_output, token)
        return 55

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_3_outputs, tokens)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_3_output, position):
        key = (attn_0_3_output, position)
        if key in {
            ("(", 0),
            ("(", 1),
            ("(", 2),
            ("(", 3),
            ("(", 4),
            ("(", 5),
            ("(", 6),
            ("(", 7),
            ("(", 8),
            ("(", 9),
            ("(", 10),
            ("(", 11),
            ("(", 12),
            ("(", 14),
            ("(", 15),
            ("(", 16),
            ("(", 18),
            ("(", 19),
            ("(", 20),
            ("(", 21),
            ("(", 22),
            ("(", 23),
            ("(", 24),
            ("(", 25),
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
            (")", 10),
            (")", 16),
            (")", 22),
            (")", 24),
            (")", 28),
            ("<s>", 1),
            ("<s>", 10),
            ("<s>", 16),
            ("<s>", 22),
            ("<s>", 24),
            ("<s>", 28),
            ("{", 1),
            ("{", 10),
            ("{", 16),
            ("{", 22),
            ("{", 24),
            ("{", 28),
            ("}", 10),
            ("}", 16),
            ("}", 18),
            ("}", 22),
            ("}", 24),
            ("}", 28),
        }:
            return 42
        elif key in {
            (")", 1),
            (")", 2),
            (")", 3),
            (")", 4),
            (")", 5),
            (")", 6),
            (")", 7),
            (")", 8),
            (")", 12),
            (")", 13),
            (")", 14),
            (")", 17),
            (")", 27),
            (")", 30),
            (")", 31),
            (")", 32),
            (")", 33),
            (")", 34),
            (")", 35),
            (")", 36),
            (")", 37),
            (")", 38),
            (")", 39),
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
            ("<s>", 4),
            ("<s>", 5),
            ("<s>", 6),
            ("<s>", 8),
            ("<s>", 12),
            ("<s>", 13),
            ("<s>", 14),
            ("<s>", 27),
            ("}", 0),
            ("}", 1),
            ("}", 2),
            ("}", 3),
            ("}", 4),
            ("}", 5),
            ("}", 6),
            ("}", 7),
            ("}", 8),
            ("}", 11),
            ("}", 12),
            ("}", 13),
            ("}", 14),
            ("}", 15),
            ("}", 17),
            ("}", 19),
            ("}", 21),
            ("}", 26),
            ("}", 27),
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
            return 5
        elif key in {
            ("(", 13),
            (")", 0),
            (")", 26),
            (")", 29),
            ("<s>", 2),
            ("<s>", 17),
            ("}", 9),
        }:
            return 54
        return 22

    mlp_0_2_outputs = [mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_3_outputs, positions)]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_3_output):
        key = attn_0_3_output
        return 48

    mlp_0_3_outputs = [mlp_0_3(k0) for k0 in attn_0_3_outputs]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output, num_attn_0_2_output):
        key = (num_attn_0_1_output, num_attn_0_2_output)
        return 30

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_2_output, num_attn_0_3_output):
        key = (num_attn_0_2_output, num_attn_0_3_output)
        return 14

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_0_output, num_attn_0_1_output):
        key = (num_attn_0_0_output, num_attn_0_1_output)
        return 36

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(one, num_attn_0_3_output):
        key = (one, num_attn_0_3_output)
        return 11

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1) for k0, k1 in zip(ones, num_attn_0_3_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_1_output, position):
        if attn_0_1_output in {"(", ")"}:
            return position == 8
        elif attn_0_1_output in {"<s>"}:
            return position == 1
        elif attn_0_1_output in {"{"}:
            return position == 7
        elif attn_0_1_output in {"}"}:
            return position == 3

    attn_1_0_pattern = select_closest(positions, attn_0_1_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_0_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_1_output, position):
        if attn_0_1_output in {"(", ")"}:
            return position == 5
        elif attn_0_1_output in {"<s>"}:
            return position == 4
        elif attn_0_1_output in {"{", "}"}:
            return position == 6

    attn_1_1_pattern = select_closest(positions, attn_0_1_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_2_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_1_output, position):
        if attn_0_1_output in {"("}:
            return position == 1
        elif attn_0_1_output in {")"}:
            return position == 3
        elif attn_0_1_output in {"<s>"}:
            return position == 4
        elif attn_0_1_output in {"{"}:
            return position == 6
        elif attn_0_1_output in {"}"}:
            return position == 2

    attn_1_2_pattern = select_closest(positions, attn_0_1_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_1_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0, 33, 32, 39, 59, 42, 45, 47, 51, 52, 54, 55, 56, 58, 27}:
            return k_position == 7
        elif q_position in {1, 4}:
            return k_position == 1
        elif q_position in {2, 10, 29}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {5, 6}:
            return k_position == 2
        elif q_position in {9, 7}:
            return k_position == 5
        elif q_position in {
            8,
            11,
            12,
            13,
            14,
            16,
            17,
            18,
            19,
            20,
            30,
            31,
            34,
            35,
            36,
            37,
            38,
            40,
            41,
            43,
            44,
            46,
            48,
            49,
            50,
            53,
            57,
        }:
            return k_position == 6
        elif q_position in {22, 15}:
            return k_position == 8
        elif q_position in {21, 23, 24, 25, 26, 28}:
            return k_position == 9

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_0_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(mlp_0_1_output, token):
        if mlp_0_1_output in {
            0,
            1,
            3,
            4,
            7,
            9,
            10,
            13,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            23,
            24,
            27,
            28,
            31,
            32,
            37,
            40,
            41,
            42,
            44,
            45,
            46,
            47,
            52,
            53,
            55,
            56,
            57,
            59,
        }:
            return token == "{"
        elif mlp_0_1_output in {33, 2, 39, 8, 12, 50}:
            return token == ""
        elif mlp_0_1_output in {
            5,
            6,
            11,
            14,
            22,
            25,
            26,
            29,
            30,
            34,
            35,
            36,
            38,
            43,
            48,
            49,
            51,
            54,
            58,
        }:
            return token == "("

    num_attn_1_0_pattern = select(tokens, mlp_0_1_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_2_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_1_output, attn_0_2_output):
        if attn_0_1_output in {"<s>", "(", ")", "}"}:
            return attn_0_2_output == "<s>"
        elif attn_0_1_output in {"{"}:
            return attn_0_2_output == "{"

    num_attn_1_1_pattern = select(attn_0_2_outputs, attn_0_1_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_0_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_0_output, attn_0_2_output):
        if attn_0_0_output in {"{", "("}:
            return attn_0_2_output == "("
        elif attn_0_0_output in {"<s>", "}", ")"}:
            return attn_0_2_output == ""

    num_attn_1_2_pattern = select(attn_0_2_outputs, attn_0_0_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_3_output, token):
        if attn_0_3_output in {"<s>", "{", "("}:
            return token == ""
        elif attn_0_3_output in {")"}:
            return token == "("
        elif attn_0_3_output in {"}"}:
            return token == "{"

    num_attn_1_3_pattern = select(tokens, attn_0_3_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_2_output, attn_1_3_output):
        key = (attn_1_2_output, attn_1_3_output)
        if key in {
            ("(", "{"),
            ("<s>", "{"),
            ("<s>", "}"),
            ("{", "("),
            ("{", ")"),
            ("{", "<s>"),
            ("{", "{"),
            ("{", "}"),
            ("}", "{"),
        }:
            return 48
        elif key in {
            ("(", "}"),
            (")", "}"),
            ("}", "("),
            ("}", ")"),
            ("}", "<s>"),
            ("}", "}"),
        }:
            return 24
        elif key in {(")", ")")}:
            return 25
        elif key in {(")", "<s>")}:
            return 41
        elif key in {("(", "<s>"), ("<s>", "<s>")}:
            return 54
        elif key in {(")", "{")}:
            return 50
        return 53

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_1_3_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(num_mlp_0_1_output, attn_0_1_output):
        key = (num_mlp_0_1_output, attn_0_1_output)
        if key in {
            (2, ")"),
            (2, "{"),
            (2, "}"),
            (4, "{"),
            (4, "}"),
            (5, "{"),
            (7, "{"),
            (9, "{"),
            (12, "{"),
            (13, "{"),
            (17, ")"),
            (17, "{"),
            (17, "}"),
            (20, "{"),
            (21, "{"),
            (23, "{"),
            (24, "{"),
            (25, "{"),
            (26, "{"),
            (28, "{"),
            (30, "{"),
            (32, "{"),
            (32, "}"),
            (33, "{"),
            (34, "{"),
            (37, "{"),
            (39, "{"),
            (40, "{"),
            (41, "{"),
            (42, "{"),
            (44, "("),
            (44, ")"),
            (44, "<s>"),
            (44, "{"),
            (44, "}"),
            (46, ")"),
            (46, "<s>"),
            (46, "{"),
            (46, "}"),
            (47, "{"),
            (48, "{"),
            (49, "{"),
            (50, ")"),
            (50, "{"),
            (50, "}"),
            (52, "{"),
            (54, "{"),
            (56, "{"),
            (57, "("),
            (57, ")"),
            (57, "<s>"),
            (57, "{"),
            (57, "}"),
            (58, "{"),
        }:
            return 8
        return 15

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(num_mlp_0_1_outputs, attn_0_1_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_1_1_output, attn_1_2_output):
        key = (attn_1_1_output, attn_1_2_output)
        return 53

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_1_2_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_1_1_output, attn_1_2_output):
        key = (attn_1_1_output, attn_1_2_output)
        if key in {("(", ")"), (")", ")"), ("}", ")")}:
            return 44
        return 30

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_1_2_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_2_output, num_attn_1_3_output):
        key = (num_attn_0_2_output, num_attn_1_3_output)
        return 22

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_2_output, num_attn_0_1_output):
        key = (num_attn_0_2_output, num_attn_0_1_output)
        return 7

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_0_1_output, num_attn_1_0_output):
        key = (num_attn_0_1_output, num_attn_1_0_output)
        return 1

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_2_output, num_attn_0_1_output):
        key = (num_attn_1_2_output, num_attn_0_1_output)
        return 46

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(token, position):
        if token in {"{", "("}:
            return position == 3
        elif token in {")"}:
            return position == 1
        elif token in {"<s>"}:
            return position == 4
        elif token in {"}"}:
            return position == 15

    attn_2_0_pattern = select_closest(positions, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(token, position):
        if token in {"}", "{", "(", ")"}:
            return position == 7
        elif token in {"<s>"}:
            return position == 1

    attn_2_1_pattern = select_closest(positions, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_1_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(token, position):
        if token in {"{", "("}:
            return position == 1
        elif token in {"}", ")"}:
            return position == 10
        elif token in {"<s>"}:
            return position == 7

    attn_2_2_pattern = select_closest(positions, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_3_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_1_output, position):
        if attn_0_1_output in {"<s>", "{", "("}:
            return position == 1
        elif attn_0_1_output in {"}", ")"}:
            return position == 9

    attn_2_3_pattern = select_closest(positions, attn_0_1_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_3_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_0_2_output, attn_0_3_output):
        if attn_0_2_output in {"<s>", "{", "("}:
            return attn_0_3_output == ""
        elif attn_0_2_output in {")"}:
            return attn_0_3_output == "("
        elif attn_0_2_output in {"}"}:
            return attn_0_3_output == "{"

    num_attn_2_0_pattern = select(attn_0_3_outputs, attn_0_2_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_3_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_2_output, attn_1_1_output):
        if attn_1_2_output in {"}", "("}:
            return attn_1_1_output == "}"
        elif attn_1_2_output in {"<s>", "{", ")"}:
            return attn_1_1_output == ")"

    num_attn_2_1_pattern = select(attn_1_1_outputs, attn_1_2_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_3_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_2_output, attn_1_3_output):
        if attn_1_2_output in {"("}:
            return attn_1_3_output == "<pad>"
        elif attn_1_2_output in {"<s>", "{", ")"}:
            return attn_1_3_output == ""
        elif attn_1_2_output in {"}"}:
            return attn_1_3_output == "{"

    num_attn_2_2_pattern = select(attn_1_3_outputs, attn_1_2_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_3_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(mlp_0_1_output, attn_0_1_output):
        if mlp_0_1_output in {
            0,
            5,
            9,
            11,
            14,
            17,
            18,
            19,
            20,
            21,
            22,
            24,
            26,
            27,
            29,
            31,
            34,
            36,
            38,
            39,
            42,
            48,
            51,
            58,
        }:
            return attn_0_1_output == "{"
        elif mlp_0_1_output in {
            1,
            3,
            4,
            6,
            7,
            8,
            10,
            12,
            13,
            15,
            16,
            25,
            28,
            30,
            32,
            33,
            35,
            37,
            40,
            41,
            43,
            44,
            45,
            46,
            47,
            49,
            50,
            52,
            53,
            54,
            55,
            56,
            57,
            59,
        }:
            return attn_0_1_output == "("
        elif mlp_0_1_output in {2}:
            return attn_0_1_output == ""
        elif mlp_0_1_output in {23}:
            return attn_0_1_output == "<s>"

    num_attn_2_3_pattern = select(attn_0_1_outputs, mlp_0_1_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_0_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_1_output, attn_2_2_output):
        key = (attn_2_1_output, attn_2_2_output)
        if key in {("<s>", "{")}:
            return 29
        elif key in {("(", "{")}:
            return 32
        return 19

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_1_outputs, attn_2_2_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(num_mlp_1_2_output, num_mlp_0_3_output):
        key = (num_mlp_1_2_output, num_mlp_0_3_output)
        return 17

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(num_mlp_1_2_outputs, num_mlp_0_3_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(mlp_0_1_output, attn_2_2_output):
        key = (mlp_0_1_output, attn_2_2_output)
        return 56

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(mlp_0_1_outputs, attn_2_2_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(num_mlp_0_0_output, attn_2_2_output):
        key = (num_mlp_0_0_output, attn_2_2_output)
        if key in {
            (0, ")"),
            (0, "<s>"),
            (0, "{"),
            (0, "}"),
            (8, "{"),
            (10, "{"),
            (18, ")"),
            (18, "<s>"),
            (18, "{"),
            (18, "}"),
            (19, ")"),
            (19, "<s>"),
            (19, "{"),
            (19, "}"),
            (27, ")"),
            (27, "<s>"),
            (27, "{"),
            (27, "}"),
            (32, ")"),
            (32, "<s>"),
            (32, "{"),
            (32, "}"),
            (46, "{"),
            (54, ")"),
            (54, "<s>"),
            (54, "{"),
            (54, "}"),
        }:
            return 7
        elif key in {
            (9, "("),
            (12, "("),
            (14, "("),
            (17, "("),
            (20, "("),
            (42, "("),
            (47, "("),
            (49, "("),
        }:
            return 41
        elif key in {
            (8, ")"),
            (8, "<s>"),
            (10, ")"),
            (10, "<s>"),
            (11, "{"),
            (23, "{"),
            (46, ")"),
            (46, "<s>"),
            (57, "{"),
        }:
            return 1
        return 17

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(num_mlp_0_0_outputs, attn_2_2_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_1_output, num_attn_2_0_output):
        key = (num_attn_1_1_output, num_attn_2_0_output)
        return 9

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_2_0_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_0_output, num_attn_1_3_output):
        key = (num_attn_2_0_output, num_attn_1_3_output)
        return 50

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_0_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_1_1_output, num_attn_1_0_output):
        key = (num_attn_1_1_output, num_attn_1_0_output)
        return 33

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_2_3_output, num_attn_1_2_output):
        key = (num_attn_2_3_output, num_attn_1_2_output)
        return 31

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_1_2_outputs)
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
            "}",
            ")",
            "(",
            "{",
            "}",
            "{",
            "}",
            "(",
            "}",
            "{",
            ")",
            "}",
            "}",
            ")",
            "}",
            "}",
            "}",
            "{",
            "(",
            "(",
            "(",
            ")",
            "}",
            "}",
            "{",
            "(",
            "{",
            "}",
            "}",
        ]
    )
)
