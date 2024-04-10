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
        "output/length/rasp/dyck2/trainlength30/s0/dyck2_weights.csv",
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
        if q_position in {
            0,
            2,
            7,
            9,
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
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
        }:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 32
        elif q_position in {10, 4}:
            return k_position == 4
        elif q_position in {5}:
            return k_position == 3
        elif q_position in {6, 23}:
            return k_position == 5
        elif q_position in {8, 11, 12, 13, 14, 15, 16}:
            return k_position == 6
        elif q_position in {17, 29, 21}:
            return k_position == 8
        elif q_position in {18, 19, 22, 24, 25, 27}:
            return k_position == 7
        elif q_position in {20}:
            return k_position == 9
        elif q_position in {26}:
            return k_position == 10
        elif q_position in {28}:
            return k_position == 11
        elif q_position in {51}:
            return k_position == 57

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {
            0,
            31,
            33,
            34,
            35,
            37,
            38,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            50,
            51,
            52,
            53,
            55,
            56,
            57,
            58,
            59,
        }:
            return k_position == 2
        elif q_position in {1, 9}:
            return k_position == 6
        elif q_position in {2}:
            return k_position == 39
        elif q_position in {32, 3, 4, 6, 40}:
            return k_position == 3
        elif q_position in {36, 5, 39, 41, 54}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 4
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 8
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
        elif q_position in {30}:
            return k_position == 50
        elif q_position in {49}:
            return k_position == 33

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {
            0,
            10,
            17,
            21,
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
            return k_position == 6
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 3}:
            return k_position == 2
        elif q_position in {8, 9, 4, 7}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {11, 12, 13, 14, 15, 16, 23, 27}:
            return k_position == 7
        elif q_position in {25, 18, 20, 22}:
            return k_position == 8
        elif q_position in {19}:
            return k_position == 10
        elif q_position in {24, 26, 28, 29}:
            return k_position == 9

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 37}:
            return k_position == 59
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 11, 5, 7}:
            return k_position == 3
        elif q_position in {3}:
            return k_position == 21
        elif q_position in {34, 51, 4}:
            return k_position == 2
        elif q_position in {9, 6}:
            return k_position == 4
        elif q_position in {8, 25, 13}:
            return k_position == 6
        elif q_position in {10}:
            return k_position == 5
        elif q_position in {17, 21, 12, 20}:
            return k_position == 7
        elif q_position in {14, 15, 23, 24, 26, 27}:
            return k_position == 8
        elif q_position in {16, 18, 29, 22}:
            return k_position == 9
        elif q_position in {19}:
            return k_position == 11
        elif q_position in {28}:
            return k_position == 10
        elif q_position in {30}:
            return k_position == 56
        elif q_position in {31}:
            return k_position == 35
        elif q_position in {32, 58, 44}:
            return k_position == 52
        elif q_position in {33}:
            return k_position == 48
        elif q_position in {59, 42, 35}:
            return k_position == 32
        elif q_position in {36, 54}:
            return k_position == 45
        elif q_position in {52, 38}:
            return k_position == 31
        elif q_position in {39}:
            return k_position == 29
        elif q_position in {40}:
            return k_position == 57
        elif q_position in {41}:
            return k_position == 44
        elif q_position in {43}:
            return k_position == 34
        elif q_position in {45}:
            return k_position == 50
        elif q_position in {49, 46}:
            return k_position == 55
        elif q_position in {47}:
            return k_position == 49
        elif q_position in {48}:
            return k_position == 54
        elif q_position in {50}:
            return k_position == 41
        elif q_position in {57, 53}:
            return k_position == 51
        elif q_position in {55}:
            return k_position == 36
        elif q_position in {56}:
            return k_position == 27

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"("}:
            return position == 48
        elif token in {")", "}"}:
            return position == 7
        elif token in {"<s>"}:
            return position == 4
        elif token in {"{"}:
            return position == 59

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(token, position):
        if token in {"(", "{", "<s>"}:
            return position == 12
        elif token in {")"}:
            return position == 28
        elif token in {"}"}:
            return position == 59

    num_attn_0_1_pattern = select(positions, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"("}:
            return position == 37
        elif token in {")"}:
            return position == 36
        elif token in {"<s>", "}"}:
            return position == 27
        elif token in {"{"}:
            return position == 54

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"("}:
            return position == 47
        elif token in {")", "}"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 55
        elif token in {"{"}:
            return position == 50

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_2_output, position):
        key = (attn_0_2_output, position)
        if key in {
            (")", 0),
            (")", 1),
            (")", 2),
            (")", 3),
            (")", 4),
            (")", 5),
            (")", 7),
            (")", 8),
            (")", 10),
            (")", 15),
            (")", 17),
            (")", 20),
            (")", 21),
            (")", 22),
            (")", 24),
            (")", 26),
            (")", 28),
            (")", 29),
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
            ("}", 0),
            ("}", 1),
            ("}", 2),
            ("}", 3),
            ("}", 4),
            ("}", 5),
            ("}", 7),
            ("}", 8),
            ("}", 10),
            ("}", 15),
            ("}", 17),
            ("}", 20),
            ("}", 21),
            ("}", 22),
            ("}", 24),
            ("}", 26),
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
            return 2
        elif key in {
            ("(", 2),
            ("(", 6),
            (")", 6),
            (")", 12),
            ("<s>", 2),
            ("<s>", 6),
            ("{", 2),
            ("{", 6),
            ("}", 6),
            ("}", 12),
        }:
            return 26
        return 21

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, positions)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_0_output, attn_0_3_output):
        key = (attn_0_0_output, attn_0_3_output)
        if key in {(")", "{"), ("<s>", "{"), ("{", "{")}:
            return 24
        return 38

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_3_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_3_output, attn_0_0_output):
        key = (attn_0_3_output, attn_0_0_output)
        return 45

    mlp_0_2_outputs = [
        mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_0_outputs)
    ]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_2_output, attn_0_3_output):
        key = (attn_0_2_output, attn_0_3_output)
        if key in {
            ("(", "("),
            ("(", "<s>"),
            ("(", "{"),
            ("(", "}"),
            (")", "<s>"),
            (")", "{"),
            ("<s>", "("),
            ("<s>", "<s>"),
            ("<s>", "{"),
            ("{", "("),
            ("{", "<s>"),
            ("{", "{"),
            ("}", "("),
            ("}", "<s>"),
        }:
            return 1
        elif key in {
            ("(", ")"),
            (")", "}"),
            ("<s>", "}"),
            ("{", "}"),
            ("}", ")"),
            ("}", "}"),
        }:
            return 19
        return 4

    mlp_0_3_outputs = [
        mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_3_outputs)
    ]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output, one):
        key = (num_attn_0_0_output, one)
        return 45

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1) for k0, k1 in zip(num_attn_0_0_outputs, ones)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_2_output):
        key = num_attn_0_2_output
        return 53

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_2_output, num_attn_0_3_output):
        key = (num_attn_0_2_output, num_attn_0_3_output)
        return 6

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_2_output, num_attn_0_3_output):
        key = (num_attn_0_2_output, num_attn_0_3_output)
        return 28

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_1_output, position):
        if attn_0_1_output in {"(", "{"}:
            return position == 9
        elif attn_0_1_output in {")"}:
            return position == 4
        elif attn_0_1_output in {"<s>"}:
            return position == 1
        elif attn_0_1_output in {"}"}:
            return position == 3

    attn_1_0_pattern = select_closest(positions, attn_0_1_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_2_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_1_output, position):
        if attn_0_1_output in {"(", "{", "<s>"}:
            return position == 6
        elif attn_0_1_output in {")", "}"}:
            return position == 5

    attn_1_1_pattern = select_closest(positions, attn_0_1_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_2_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_1_output, position):
        if attn_0_1_output in {"(", "{", "<s>"}:
            return position == 4
        elif attn_0_1_output in {")"}:
            return position == 5
        elif attn_0_1_output in {"}"}:
            return position == 6

    attn_1_2_pattern = select_closest(positions, attn_0_1_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_3_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_1_output, position):
        if attn_0_1_output in {"(", "{"}:
            return position == 9
        elif attn_0_1_output in {")"}:
            return position == 3
        elif attn_0_1_output in {"<s>"}:
            return position == 1
        elif attn_0_1_output in {"}"}:
            return position == 4

    attn_1_3_pattern = select_closest(positions, attn_0_1_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_2_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(mlp_0_1_output, attn_0_1_output):
        if mlp_0_1_output in {0, 32, 2, 3, 5, 8, 12, 44, 47, 17, 52, 22, 23, 25}:
            return attn_0_1_output == ""
        elif mlp_0_1_output in {
            1,
            9,
            13,
            15,
            16,
            20,
            24,
            27,
            30,
            31,
            33,
            34,
            36,
            38,
            39,
            45,
            46,
            49,
            57,
        }:
            return attn_0_1_output == "{"
        elif mlp_0_1_output in {
            4,
            6,
            7,
            10,
            11,
            14,
            18,
            19,
            21,
            26,
            28,
            29,
            35,
            37,
            40,
            41,
            42,
            43,
            48,
            50,
            51,
            53,
            54,
            55,
            56,
            58,
            59,
        }:
            return attn_0_1_output == "("

    num_attn_1_0_pattern = select(attn_0_1_outputs, mlp_0_1_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_1_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(mlp_0_1_output, attn_0_1_output):
        if mlp_0_1_output in {0, 41}:
            return attn_0_1_output == "{"
        elif mlp_0_1_output in {
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
            14,
            15,
            17,
            19,
            20,
            26,
            28,
            30,
            31,
            33,
            34,
            35,
            37,
            39,
            40,
            42,
            43,
            44,
            49,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
        }:
            return attn_0_1_output == ""
        elif mlp_0_1_output in {32, 2, 36, 16, 50, 51, 22, 24}:
            return attn_0_1_output == ")"
        elif mlp_0_1_output in {5, 38, 45, 47, 21, 23, 25, 29}:
            return attn_0_1_output == "}"
        elif mlp_0_1_output in {48, 18, 52, 13}:
            return attn_0_1_output == "("
        elif mlp_0_1_output in {27}:
            return attn_0_1_output == "<s>"
        elif mlp_0_1_output in {46}:
            return attn_0_1_output == "<pad>"

    num_attn_1_1_pattern = select(attn_0_1_outputs, mlp_0_1_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_3_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(mlp_0_1_output, attn_0_0_output):
        if mlp_0_1_output in {
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
            19,
            20,
            21,
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
            39,
            40,
            42,
            43,
            48,
            49,
            51,
            52,
            54,
            56,
            57,
            58,
        }:
            return attn_0_0_output == ""
        elif mlp_0_1_output in {32, 2, 5, 41, 44, 45, 46, 50, 22, 23, 24, 25, 31}:
            return attn_0_0_output == "("
        elif mlp_0_1_output in {38, 14, 47, 53, 59}:
            return attn_0_0_output == "{"
        elif mlp_0_1_output in {55}:
            return attn_0_0_output == "<s>"

    num_attn_1_2_pattern = select(attn_0_0_outputs, mlp_0_1_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_1_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_2_output, attn_0_1_output):
        if attn_0_2_output in {"(", "{", "<s>"}:
            return attn_0_1_output == ""
        elif attn_0_2_output in {")"}:
            return attn_0_1_output == "("
        elif attn_0_2_output in {"}"}:
            return attn_0_1_output == "{"

    num_attn_1_3_pattern = select(attn_0_1_outputs, attn_0_2_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_0_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_3_output, attn_1_0_output):
        key = (attn_1_3_output, attn_1_0_output)
        if key in {(")", ")"), (")", "<s>"), ("<s>", ")"), ("{", ")"), ("}", ")")}:
            return 54
        return 37

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_3_outputs, attn_1_0_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_2_output, attn_1_3_output):
        key = (attn_1_2_output, attn_1_3_output)
        if key in {("(", "}"), (")", "}"), ("}", "("), ("}", ")")}:
            return 40
        elif key in {("(", "{"), (")", "{"), ("{", "("), ("{", "<s>"), ("{", "{")}:
            return 32
        elif key in {("<s>", "}"), ("}", "<s>"), ("}", "}")}:
            return 36
        elif key in {("<s>", "{"), ("{", "}"), ("}", "{")}:
            return 24
        return 54

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_1_3_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_1_0_output, position):
        key = (attn_1_0_output, position)
        if key in {
            (")", 0),
            (")", 1),
            (")", 3),
            (")", 4),
            (")", 8),
            (")", 10),
            (")", 11),
            (")", 12),
            (")", 13),
            (")", 14),
            (")", 15),
            (")", 16),
            (")", 17),
            (")", 18),
            (")", 19),
            (")", 20),
            (")", 21),
            (")", 22),
            (")", 23),
            (")", 24),
            (")", 25),
            (")", 26),
            (")", 27),
            (")", 28),
            (")", 29),
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
        }:
            return 19
        return 54

    mlp_1_2_outputs = [mlp_1_2(k0, k1) for k0, k1 in zip(attn_1_0_outputs, positions)]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_0_2_output, attn_1_0_output):
        key = (attn_0_2_output, attn_1_0_output)
        return 20

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_1_0_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_1_output):
        key = num_attn_0_1_output
        return 12

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_0_1_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_0_output, num_attn_1_3_output):
        key = (num_attn_1_0_output, num_attn_1_3_output)
        return 28

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_0_output, num_attn_1_1_output):
        key = (num_attn_1_0_output, num_attn_1_1_output)
        return 46

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_2_output, num_attn_0_3_output):
        key = (num_attn_1_2_output, num_attn_0_3_output)
        return 22

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(token, position):
        if token in {"("}:
            return position == 4
        elif token in {")"}:
            return position == 9
        elif token in {"<s>"}:
            return position == 6
        elif token in {"{"}:
            return position == 7
        elif token in {"}"}:
            return position == 10

    attn_2_0_pattern = select_closest(positions, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_2_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(token, position):
        if token in {"(", "<s>"}:
            return position == 6
        elif token in {")", "}", "{"}:
            return position == 7

    attn_2_1_pattern = select_closest(positions, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_1_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(token, position):
        if token in {"(", ")", "{"}:
            return position == 1
        elif token in {"<s>"}:
            return position == 7
        elif token in {"}"}:
            return position == 10

    attn_2_2_pattern = select_closest(positions, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_3_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(token, position):
        if token in {"(", "}", "{"}:
            return position == 1
        elif token in {")"}:
            return position == 13
        elif token in {"<s>"}:
            return position == 7

    attn_2_3_pattern = select_closest(positions, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(num_mlp_0_0_output, attn_1_2_output):
        if num_mlp_0_0_output in {
            0,
            4,
            5,
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
            48,
            49,
            51,
            53,
            54,
            55,
            57,
            58,
            59,
        }:
            return attn_1_2_output == ""
        elif num_mlp_0_0_output in {1, 3}:
            return attn_1_2_output == "{"
        elif num_mlp_0_0_output in {2}:
            return attn_1_2_output == "("
        elif num_mlp_0_0_output in {32, 7, 40, 47, 50, 52, 23, 56}:
            return attn_1_2_output == "<s>"

    num_attn_2_0_pattern = select(
        attn_1_2_outputs, num_mlp_0_0_outputs, num_predicate_2_0
    )
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(num_mlp_0_3_output, attn_1_0_output):
        if num_mlp_0_3_output in {
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
            57,
            58,
            59,
        }:
            return attn_1_0_output == ""
        elif num_mlp_0_3_output in {56, 27, 13, 39}:
            return attn_1_0_output == "<s>"
        elif num_mlp_0_3_output in {43}:
            return attn_1_0_output == "<pad>"

    num_attn_2_1_pattern = select(
        attn_1_0_outputs, num_mlp_0_3_outputs, num_predicate_2_1
    )
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_3_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_0_2_output, attn_1_2_output):
        if attn_0_2_output in {"(", "}"}:
            return attn_1_2_output == "{"
        elif attn_0_2_output in {"<s>", ")", "{"}:
            return attn_1_2_output == "("

    num_attn_2_2_pattern = select(attn_1_2_outputs, attn_0_2_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_3_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(mlp_1_1_output, attn_1_2_output):
        if mlp_1_1_output in {
            0,
            2,
            3,
            4,
            7,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            21,
            23,
            25,
            27,
            28,
            32,
            33,
            42,
            43,
            45,
            46,
            48,
            51,
            52,
            54,
            56,
            57,
        }:
            return attn_1_2_output == "("
        elif mlp_1_1_output in {
            1,
            5,
            6,
            8,
            18,
            19,
            20,
            22,
            26,
            29,
            34,
            35,
            36,
            37,
            38,
            40,
            41,
            44,
            47,
            50,
            53,
            55,
            58,
            59,
        }:
            return attn_1_2_output == "{"
        elif mlp_1_1_output in {16, 39}:
            return attn_1_2_output == "<s>"
        elif mlp_1_1_output in {17, 49, 24, 30, 31}:
            return attn_1_2_output == ""

    num_attn_2_3_pattern = select(attn_1_2_outputs, mlp_1_1_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_3_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_2_output, attn_2_0_output):
        key = (attn_2_2_output, attn_2_0_output)
        if key in {
            ("(", ")"),
            (")", ")"),
            (")", "<s>"),
            ("<s>", ")"),
            ("<s>", "<s>"),
            ("{", ")"),
            ("{", "<s>"),
            ("}", ")"),
            ("}", "<s>"),
        }:
            return 53
        elif key in {(")", "{")}:
            return 44
        elif key in {(")", "}")}:
            return 39
        return 33

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_2_outputs, attn_2_0_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_0_3_output, attn_1_0_output):
        key = (attn_0_3_output, attn_1_0_output)
        if key in {
            ("(", "<s>"),
            ("(", "{"),
            (")", "("),
            (")", "<s>"),
            (")", "{"),
            ("<s>", "<s>"),
            ("<s>", "{"),
            ("{", "("),
            ("{", "<s>"),
            ("{", "{"),
            ("}", "<s>"),
            ("}", "{"),
        }:
            return 16
        elif key in {("<s>", "(")}:
            return 51
        elif key in {("(", "("), ("}", "(")}:
            return 57
        return 41

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_1_0_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(attn_2_1_output, attn_2_0_output):
        key = (attn_2_1_output, attn_2_0_output)
        if key in {
            ("(", "("),
            ("(", "<s>"),
            ("(", "{"),
            ("{", "("),
            ("{", "<s>"),
            ("{", "{"),
        }:
            return 43
        elif key in {
            ("(", ")"),
            ("(", "}"),
            (")", "("),
            (")", ")"),
            (")", "<s>"),
            (")", "{"),
            (")", "}"),
            ("<s>", ")"),
            ("<s>", "}"),
            ("{", ")"),
            ("{", "}"),
            ("}", "("),
            ("}", ")"),
            ("}", "<s>"),
            ("}", "{"),
            ("}", "}"),
        }:
            return 22
        return 41

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(attn_2_1_outputs, attn_2_0_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(attn_1_0_output, attn_1_1_output):
        key = (attn_1_0_output, attn_1_1_output)
        return 53

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(attn_1_0_outputs, attn_1_1_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_1_output, num_attn_2_0_output):
        key = (num_attn_2_1_output, num_attn_2_0_output)
        return 18

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_2_0_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_0_output, num_attn_1_2_output):
        key = (num_attn_1_0_output, num_attn_1_2_output)
        return 14

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_1_0_output, num_attn_2_1_output):
        key = (num_attn_1_0_output, num_attn_2_1_output)
        return 6

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_1_2_output, num_attn_1_1_output):
        key = (num_attn_1_2_output, num_attn_1_1_output)
        return 11

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
            "}",
            ")",
            "(",
            "}",
            "}",
            "}",
            "}",
            ")",
            "}",
            ")",
            "{",
            "(",
            "}",
            "{",
            "(",
            "(",
            "(",
            "{",
            ")",
            "{",
            "}",
            "}",
            "{",
            "(",
            ")",
            ")",
            ")",
            ")",
            "(",
        ]
    )
)
