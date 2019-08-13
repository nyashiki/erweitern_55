#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
    NoColor,
}

impl Color {
    pub fn get_op_color(self) -> Color {
        if self == Color::White {
            Color::Black
        } else {
            Color::White
        }
    }
}

#[test]
fn get_op_color_test() {
    assert!(Color::White.get_op_color() == Color::Black);
    assert!(Color::Black.get_op_color() == Color::White);
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Piece {
    NoPiece = 0,

    WKing = 0b00001,
    WGold = 0b00010,
    WSilver = 0b00011,
    WBishop = 0b00100,
    WRook = 0b00101,
    WPawn = 0b00110,
    WSilverX = 0b01011,
    WBishopX = 0b01100,
    WRookX = 0b01101,
    WPawnX = 0b01110,

    BKing = 0b10001,
    BGold = 0b10010,
    BSilver = 0b10011,
    BBishop = 0b10100,
    BRook = 0b10101,
    BPawn = 0b10110,
    BSilverX = 0b11011,
    BBishopX = 0b11100,
    BRookX = 0b11101,
    BPawnX = 0b11110,
}

impl Piece {
    pub fn get_promoted(self) -> Piece {
        match self {
            Piece::WSilver => Piece::WSilverX,
            Piece::WBishop => Piece::WBishopX,
            Piece::WRook => Piece::WRookX,
            Piece::WPawn => Piece::WPawnX,
            Piece::BSilver => Piece::BSilverX,
            Piece::BBishop => Piece::BBishopX,
            Piece::BRook => Piece::BRookX,
            Piece::BPawn => Piece::BPawnX,
            _ => Piece::NoPiece,
        }
    }

    pub fn is_promoted(self) -> bool {
        match self {
            Piece::WSilverX => true,
            Piece::WBishopX => true,
            Piece::WRookX => true,
            Piece::WPawnX => true,
            Piece::BSilverX => true,
            Piece::BBishopX => true,
            Piece::BRookX => true,
            Piece::BPawnX => true,
            _ => false,
        }
    }

    pub fn is_promotable(self) -> bool {
        return self.get_promoted() != Piece::NoPiece;
    }

    pub fn get_raw(self) -> Piece {
        if !self.is_promoted() {
            return self;
        }

        match self {
            Piece::WSilverX => Piece::WSilver,
            Piece::WBishopX => Piece::WBishop,
            Piece::WRookX => Piece::WRook,
            Piece::WPawnX => Piece::WPawn,
            Piece::BSilverX => Piece::BSilver,
            Piece::BBishopX => Piece::BBishop,
            Piece::BRookX => Piece::BRook,
            Piece::BPawnX => Piece::BPawn,
            _ => Piece::NoPiece,
        }
    }

    pub fn is_raw(self) -> bool {
        !self.is_promoted()
    }

    pub fn get_color(self) -> Color {
        match self {
            Piece::NoPiece => Color::NoColor,

            Piece::WKing => Color::White,
            Piece::WGold => Color::White,
            Piece::WSilver => Color::White,
            Piece::WBishop => Color::White,
            Piece::WRook => Color::White,
            Piece::WPawn => Color::White,
            Piece::WSilverX => Color::White,
            Piece::WBishopX => Color::White,
            Piece::WRookX => Color::White,
            Piece::WPawnX => Color::White,

            Piece::BKing => Color::Black,
            Piece::BGold => Color::Black,
            Piece::BSilver => Color::Black,
            Piece::BBishop => Color::Black,
            Piece::BRook => Color::Black,
            Piece::BPawn => Color::Black,
            Piece::BSilverX => Color::Black,
            Piece::BBishopX => Color::Black,
            Piece::BRookX => Color::Black,
            Piece::BPawnX => Color::Black,
        }
    }

    pub fn get_piece_type(self) -> PieceType {
        match self {
            Piece::NoPiece => PieceType::NoPieceType,

            Piece::WKing => PieceType::King,
            Piece::WGold => PieceType::Gold,
            Piece::WSilver => PieceType::Silver,
            Piece::WBishop => PieceType::Bishop,
            Piece::WRook => PieceType::Rook,
            Piece::WPawn => PieceType::Pawn,
            Piece::WSilverX => PieceType::SilverX,
            Piece::WBishopX => PieceType::BishopX,
            Piece::WRookX => PieceType::RookX,
            Piece::WPawnX => PieceType::PawnX,

            Piece::BKing => PieceType::King,
            Piece::BGold => PieceType::Gold,
            Piece::BSilver => PieceType::Silver,
            Piece::BBishop => PieceType::Bishop,
            Piece::BRook => PieceType::Rook,
            Piece::BPawn => PieceType::Pawn,
            Piece::BSilverX => PieceType::SilverX,
            Piece::BBishopX => PieceType::BishopX,
            Piece::BRookX => PieceType::RookX,
            Piece::BPawnX => PieceType::PawnX,
        }
    }

    pub fn get_op_piece(self) -> Piece {
        match self {
            Piece::NoPiece => Piece::NoPiece,

            Piece::WKing => Piece::BKing,
            Piece::WGold => Piece::BGold,
            Piece::WSilver => Piece::BSilver,
            Piece::WBishop => Piece::BBishop,
            Piece::WRook => Piece::BRook,
            Piece::WPawn => Piece::BPawn,
            Piece::WSilverX => Piece::BSilverX,
            Piece::WBishopX => Piece::BBishopX,
            Piece::WRookX => Piece::BRookX,
            Piece::WPawnX => Piece::BPawnX,

            Piece::BKing => Piece::WKing,
            Piece::BGold => Piece::WGold,
            Piece::BSilver => Piece::WSilver,
            Piece::BBishop => Piece::WBishop,
            Piece::BRook => Piece::WRook,
            Piece::BPawn => Piece::WPawn,
            Piece::BSilverX => Piece::WSilverX,
            Piece::BBishopX => Piece::WBishopX,
            Piece::BRookX => Piece::WRookX,
            Piece::BPawnX => Piece::WPawnX,
        }
    }

    pub fn get_move_dirs(self) -> std::vec::Vec<Direction> {
        match self {
            Piece::WKing => vec![
                Direction::N,
                Direction::NE,
                Direction::E,
                Direction::SE,
                Direction::S,
                Direction::SW,
                Direction::W,
                Direction::NW,
            ],
            Piece::WGold => vec![
                Direction::N,
                Direction::NE,
                Direction::E,
                Direction::S,
                Direction::W,
                Direction::NW,
            ],
            Piece::WSilver => {
                vec![Direction::N, Direction::NE, Direction::SE, Direction::SW, Direction::NW]
            }
            Piece::WPawn => vec![Direction::N],
            Piece::WSilverX => vec![
                Direction::N,
                Direction::NE,
                Direction::E,
                Direction::S,
                Direction::W,
                Direction::NW,
            ],
            Piece::WBishopX => vec![Direction::N, Direction::E, Direction::S, Direction::W],
            Piece::WRookX => vec![Direction::NE, Direction::SE, Direction::SW, Direction::NW],
            Piece::WPawnX => vec![
                Direction::N,
                Direction::NE,
                Direction::E,
                Direction::S,
                Direction::W,
                Direction::NW,
            ],

            Piece::BKing => vec![
                Direction::N,
                Direction::NE,
                Direction::E,
                Direction::SE,
                Direction::S,
                Direction::SW,
                Direction::W,
                Direction::NW,
            ],
            Piece::BGold => vec![
                Direction::N,
                Direction::E,
                Direction::SE,
                Direction::S,
                Direction::SW,
                Direction::W,
            ],
            Piece::BSilver => {
                vec![Direction::NE, Direction::SE, Direction::S, Direction::SW, Direction::NW]
            }
            Piece::BPawn => vec![Direction::S],
            Piece::BSilverX => vec![
                Direction::N,
                Direction::E,
                Direction::SE,
                Direction::S,
                Direction::SW,
                Direction::W,
            ],
            Piece::BBishopX => vec![Direction::N, Direction::E, Direction::S, Direction::W],
            Piece::BRookX => vec![Direction::NE, Direction::SE, Direction::SW, Direction::NW],
            Piece::BPawnX => vec![
                Direction::N,
                Direction::E,
                Direction::SE,
                Direction::S,
                Direction::SW,
                Direction::W,
            ],

            _ => vec![],
        }
    }
}

impl std::fmt::Display for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Piece::NoPiece => write!(f, " * "),

            Piece::WKing => write!(f, "\x1b[38;2;0;100;200m K \x1b[0m"),
            Piece::WGold => write!(f, "\x1b[38;2;0;100;200m G \x1b[0m"),
            Piece::WSilver => write!(f, "\x1b[38;2;0;100;200m S \x1b[0m"),
            Piece::WBishop => write!(f, "\x1b[38;2;0;100;200m B \x1b[0m"),
            Piece::WRook => write!(f, "\x1b[38;2;0;100;200m R \x1b[0m"),
            Piece::WPawn => write!(f, "\x1b[38;2;0;100;200m P \x1b[0m"),
            Piece::WSilverX => write!(f, "\x1b[38;2;0;100;200m Sx\x1b[0m"),
            Piece::WBishopX => write!(f, "\x1b[38;2;0;100;200m Bx\x1b[0m"),
            Piece::WRookX => write!(f, "\x1b[38;2;0;100;200m Rx\x1b[0m"),
            Piece::WPawnX => write!(f, "\x1b[38;2;0;100;200m Px\x1b[0m"),

            Piece::BKing => write!(f, "\x1b[38;2;250;200;50mvK \x1b[0m"),
            Piece::BGold => write!(f, "\x1b[38;2;250;200;50mvG \x1b[0m"),
            Piece::BSilver => write!(f, "\x1b[38;2;250;200;50mvS \x1b[0m"),
            Piece::BBishop => write!(f, "\x1b[38;2;250;200;50mvB \x1b[0m"),
            Piece::BRook => write!(f, "\x1b[38;2;250;200;50mvR \x1b[0m"),
            Piece::BPawn => write!(f, "\x1b[38;2;250;200;50mvP \x1b[0m"),
            Piece::BSilverX => write!(f, "\x1b[38;2;250;200;50mvSx\x1b[0m"),
            Piece::BBishopX => write!(f, "\x1b[38;2;250;200;50mvBx\x1b[0m"),
            Piece::BRookX => write!(f, "\x1b[38;2;250;200;50mvRx\x1b[0m"),
            Piece::BPawnX => write!(f, "\x1b[38;2;250;200;50mvPx\x1b[0m"),
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
#[repr(u8)]
pub enum PieceType {
    NoPieceType = 0,

    King = 0b0001,
    Gold = 0b0010,
    Silver = 0b0011,
    Bishop = 0b0100,
    Rook = 0b0101,
    Pawn = 0b0110,
    SilverX = 0b1011,
    BishopX = 0b1100,
    RookX = 0b1101,
    PawnX = 0b1110,
}

impl PieceType {
    pub fn get_promoted(self) -> PieceType {
        match self {
            PieceType::Silver => PieceType::SilverX,
            PieceType::Bishop => PieceType::BishopX,
            PieceType::Rook => PieceType::RookX,
            PieceType::Pawn => PieceType::PawnX,
            _ => PieceType::NoPieceType,
        }
    }

    pub fn is_promoted(self) -> bool {
        match self {
            PieceType::SilverX => true,
            PieceType::BishopX => true,
            PieceType::RookX => true,
            PieceType::PawnX => true,
            _ => false,
        }
    }

    pub fn is_promotable(self) -> bool {
        return self.get_promoted() != PieceType::NoPieceType;
    }

    pub fn get_raw(self) -> PieceType {
        if !self.is_promoted() {
            return self;
        }

        match self {
            PieceType::SilverX => PieceType::Silver,
            PieceType::BishopX => PieceType::Bishop,
            PieceType::RookX => PieceType::Rook,
            PieceType::PawnX => PieceType::Pawn,
            _ => PieceType::NoPieceType,
        }
    }

    pub fn is_raw(self) -> bool {
        !self.is_promoted()
    }

    pub fn get_piece(self, color: Color) -> Piece {
        if color == Color::White {
            match self {
                PieceType::King => Piece::WKing,
                PieceType::Gold => Piece::WGold,
                PieceType::Silver => Piece::WSilver,
                PieceType::Bishop => Piece::WBishop,
                PieceType::Rook => Piece::WRook,
                PieceType::Pawn => Piece::WPawn,
                PieceType::SilverX => Piece::WSilverX,
                PieceType::BishopX => Piece::WBishopX,
                PieceType::RookX => Piece::WRookX,
                PieceType::PawnX => Piece::WPawnX,
                _ => Piece::NoPiece,
            }
        } else {
            match self {
                PieceType::King => Piece::BKing,
                PieceType::Gold => Piece::BGold,
                PieceType::Silver => Piece::BSilver,
                PieceType::Bishop => Piece::BBishop,
                PieceType::Rook => Piece::BRook,
                PieceType::Pawn => Piece::BPawn,
                PieceType::SilverX => Piece::BSilverX,
                PieceType::BishopX => Piece::BBishopX,
                PieceType::RookX => Piece::BRookX,
                PieceType::PawnX => Piece::BPawnX,
                _ => Piece::NoPiece,
            }
        }
    }
}

#[test]
fn get_promoted_test() {
    // Piece
    assert!(Piece::WSilver.get_promoted() == Piece::WSilverX);
    assert!(Piece::WBishop.get_promoted() == Piece::WBishopX);
    assert!(Piece::WRook.get_promoted() == Piece::WRookX);
    assert!(Piece::WPawn.get_promoted() == Piece::WPawnX);
    assert!(Piece::BSilver.get_promoted() == Piece::BSilverX);
    assert!(Piece::BBishop.get_promoted() == Piece::BBishopX);
    assert!(Piece::BRook.get_promoted() == Piece::BRookX);
    assert!(Piece::BPawn.get_promoted() == Piece::BPawnX);

    // PieceType
    assert!(PieceType::Silver.get_promoted() == PieceType::SilverX);
    assert!(PieceType::Bishop.get_promoted() == PieceType::BishopX);
    assert!(PieceType::Rook.get_promoted() == PieceType::RookX);
    assert!(PieceType::Pawn.get_promoted() == PieceType::PawnX);
}

#[test]
fn is_promoted_test() {
    // Piece
    assert!(!Piece::WKing.is_promoted());
    assert!(!Piece::WGold.is_promoted());
    assert!(!Piece::WSilver.is_promoted());
    assert!(!Piece::WBishop.is_promoted());
    assert!(!Piece::WRook.is_promoted());
    assert!(!Piece::WPawn.is_promoted());
    assert!(Piece::WSilverX.is_promoted());
    assert!(Piece::WBishopX.is_promoted());
    assert!(Piece::WRookX.is_promoted());
    assert!(Piece::WPawnX.is_promoted());
    assert!(!Piece::BKing.is_promoted());
    assert!(!Piece::BGold.is_promoted());
    assert!(!Piece::BSilver.is_promoted());
    assert!(!Piece::BBishop.is_promoted());
    assert!(!Piece::BRook.is_promoted());
    assert!(!Piece::BPawn.is_promoted());
    assert!(Piece::BSilverX.is_promoted());
    assert!(Piece::BBishopX.is_promoted());
    assert!(Piece::BRookX.is_promoted());
    assert!(Piece::BPawnX.is_promoted());

    // PieceType
    assert!(!PieceType::King.is_promoted());
    assert!(!PieceType::Gold.is_promoted());
    assert!(!PieceType::Silver.is_promoted());
    assert!(!PieceType::Bishop.is_promoted());
    assert!(!PieceType::Rook.is_promoted());
    assert!(!PieceType::Pawn.is_promoted());
    assert!(PieceType::SilverX.is_promoted());
    assert!(PieceType::BishopX.is_promoted());
    assert!(PieceType::RookX.is_promoted());
    assert!(PieceType::PawnX.is_promoted());
}

#[test]
fn get_raw_test() {
    // Piece
    assert!(Piece::WSilverX.get_raw() == Piece::WSilver);
    assert!(Piece::WBishopX.get_raw() == Piece::WBishop);
    assert!(Piece::WRookX.get_raw() == Piece::WRook);
    assert!(Piece::WPawnX.get_raw() == Piece::WPawn);
    assert!(Piece::BSilverX.get_raw() == Piece::BSilver);
    assert!(Piece::BBishopX.get_raw() == Piece::BBishop);
    assert!(Piece::BRookX.get_raw() == Piece::BRook);
    assert!(Piece::BPawnX.get_raw() == Piece::BPawn);

    // PieceType
    assert!(PieceType::SilverX.get_raw() == PieceType::Silver);
    assert!(PieceType::BishopX.get_raw() == PieceType::Bishop);
    assert!(PieceType::RookX.get_raw() == PieceType::Rook);
    assert!(PieceType::PawnX.get_raw() == PieceType::Pawn);
}

#[test]
fn is_raw_test() {
    // Piece
    assert!(Piece::WKing.is_raw());
    assert!(Piece::WGold.is_raw());
    assert!(Piece::WBishop.is_raw());
    assert!(Piece::WRook.is_raw());
    assert!(Piece::WPawn.is_raw());
    assert!(Piece::BKing.is_raw());
    assert!(Piece::BGold.is_raw());
    assert!(Piece::BSilver.is_raw());
    assert!(Piece::BBishop.is_raw());
    assert!(Piece::BRook.is_raw());
    assert!(Piece::BPawn.is_raw());

    // PieceType
    assert!(PieceType::King.is_raw());
    assert!(PieceType::Gold.is_raw());
    assert!(PieceType::Silver.is_raw());
    assert!(PieceType::Bishop.is_raw());
    assert!(PieceType::Rook.is_raw());
    assert!(PieceType::Pawn.is_raw());
}

#[test]
fn get_piece_test() {
    assert!(PieceType::NoPieceType.get_piece(Color::White) == Piece::NoPiece);

    // White
    assert!(PieceType::King.get_piece(Color::White) == Piece::WKing);
    assert!(PieceType::Gold.get_piece(Color::White) == Piece::WGold);
    assert!(PieceType::Silver.get_piece(Color::White) == Piece::WSilver);
    assert!(PieceType::Bishop.get_piece(Color::White) == Piece::WBishop);
    assert!(PieceType::Rook.get_piece(Color::White) == Piece::WRook);
    assert!(PieceType::Pawn.get_piece(Color::White) == Piece::WPawn);
    assert!(PieceType::SilverX.get_piece(Color::White) == Piece::WSilverX);
    assert!(PieceType::BishopX.get_piece(Color::White) == Piece::WBishopX);
    assert!(PieceType::RookX.get_piece(Color::White) == Piece::WRookX);
    assert!(PieceType::PawnX.get_piece(Color::White) == Piece::WPawnX);

    // Black
    assert!(PieceType::King.get_piece(Color::Black) == Piece::BKing);
    assert!(PieceType::Gold.get_piece(Color::Black) == Piece::BGold);
    assert!(PieceType::Silver.get_piece(Color::Black) == Piece::BSilver);
    assert!(PieceType::Bishop.get_piece(Color::Black) == Piece::BBishop);
    assert!(PieceType::Rook.get_piece(Color::Black) == Piece::BRook);
    assert!(PieceType::Pawn.get_piece(Color::Black) == Piece::BPawn);
    assert!(PieceType::SilverX.get_piece(Color::Black) == Piece::BSilverX);
    assert!(PieceType::BishopX.get_piece(Color::Black) == Piece::BBishopX);
    assert!(PieceType::RookX.get_piece(Color::Black) == Piece::BRookX);
    assert!(PieceType::PawnX.get_piece(Color::Black) == Piece::BPawnX);
}

#[test]
fn get_op_piece_test() {
    assert!(Piece::NoPiece.get_op_piece() == Piece::NoPiece);

    // White
    assert!(Piece::WKing.get_op_piece() == Piece::BKing);
    assert!(Piece::WGold.get_op_piece() == Piece::BGold);
    assert!(Piece::WSilver.get_op_piece() == Piece::BSilver);
    assert!(Piece::WBishop.get_op_piece() == Piece::BBishop);
    assert!(Piece::WRook.get_op_piece() == Piece::BRook);
    assert!(Piece::WPawn.get_op_piece() == Piece::BPawn);
    assert!(Piece::WSilverX.get_op_piece() == Piece::BSilverX);
    assert!(Piece::WBishopX.get_op_piece() == Piece::BBishopX);
    assert!(Piece::WRookX.get_op_piece() == Piece::BRookX);
    assert!(Piece::WPawnX.get_op_piece() == Piece::BPawnX);

    // Black
    assert!(Piece::BKing.get_op_piece() == Piece::WKing);
    assert!(Piece::BGold.get_op_piece() == Piece::WGold);
    assert!(Piece::BSilver.get_op_piece() == Piece::WSilver);
    assert!(Piece::BBishop.get_op_piece() == Piece::WBishop);
    assert!(Piece::BRook.get_op_piece() == Piece::WRook);
    assert!(Piece::BPawn.get_op_piece() == Piece::WPawn);
    assert!(Piece::BSilverX.get_op_piece() == Piece::WSilverX);
    assert!(Piece::BBishopX.get_op_piece() == Piece::WBishopX);
    assert!(Piece::BRookX.get_op_piece() == Piece::WRookX);
    assert!(Piece::BPawnX.get_op_piece() == Piece::WPawnX);
}

#[test]
fn get_color_test() {
    assert!(Piece::NoPiece.get_color() == Color::NoColor);

    assert!(Piece::WKing.get_color() == Color::White);
    assert!(Piece::WGold.get_color() == Color::White);
    assert!(Piece::WSilver.get_color() == Color::White);
    assert!(Piece::WBishop.get_color() == Color::White);
    assert!(Piece::WRook.get_color() == Color::White);
    assert!(Piece::WPawn.get_color() == Color::White);
    assert!(Piece::WSilverX.get_color() == Color::White);
    assert!(Piece::WBishopX.get_color() == Color::White);
    assert!(Piece::WRookX.get_color() == Color::White);
    assert!(Piece::WPawnX.get_color() == Color::White);

    assert!(Piece::BKing.get_color() == Color::Black);
    assert!(Piece::BGold.get_color() == Color::Black);
    assert!(Piece::BSilver.get_color() == Color::Black);
    assert!(Piece::BBishop.get_color() == Color::Black);
    assert!(Piece::BRook.get_color() == Color::Black);
    assert!(Piece::BPawn.get_color() == Color::Black);
    assert!(Piece::BSilverX.get_color() == Color::Black);
    assert!(Piece::BBishopX.get_color() == Color::Black);
    assert!(Piece::BRookX.get_color() == Color::Black);
    assert!(Piece::BPawnX.get_color() == Color::Black);
}

#[test]
fn get_piece_type_test() {
    assert!(Piece::NoPiece.get_piece_type() == PieceType::NoPieceType);

    assert!(Piece::WKing.get_piece_type() == PieceType::King);
    assert!(Piece::WGold.get_piece_type() == PieceType::Gold);
    assert!(Piece::WSilver.get_piece_type() == PieceType::Silver);
    assert!(Piece::WBishop.get_piece_type() == PieceType::Bishop);
    assert!(Piece::WRook.get_piece_type() == PieceType::Rook);
    assert!(Piece::WPawn.get_piece_type() == PieceType::Pawn);
    assert!(Piece::WSilverX.get_piece_type() == PieceType::SilverX);
    assert!(Piece::WBishopX.get_piece_type() == PieceType::BishopX);
    assert!(Piece::WRookX.get_piece_type() == PieceType::RookX);
    assert!(Piece::WPawnX.get_piece_type() == PieceType::PawnX);

    assert!(Piece::BKing.get_piece_type() == PieceType::King);
    assert!(Piece::BGold.get_piece_type() == PieceType::Gold);
    assert!(Piece::BSilver.get_piece_type() == PieceType::Silver);
    assert!(Piece::BBishop.get_piece_type() == PieceType::Bishop);
    assert!(Piece::BRook.get_piece_type() == PieceType::Rook);
    assert!(Piece::BPawn.get_piece_type() == PieceType::Pawn);
    assert!(Piece::BSilverX.get_piece_type() == PieceType::SilverX);
    assert!(Piece::BBishopX.get_piece_type() == PieceType::BishopX);
    assert!(Piece::BRookX.get_piece_type() == PieceType::RookX);
    assert!(Piece::BPawnX.get_piece_type() == PieceType::PawnX);
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Direction {
    N = 0,
    NE = 1,
    E = 2,
    SE = 3,
    S = 4,
    SW = 5,
    W = 6,
    NW = 7,
}

pub const PIECE_ALL: [Piece; 20] = [
    Piece::WKing,
    Piece::WGold,
    Piece::WSilver,
    Piece::WBishop,
    Piece::WRook,
    Piece::WPawn,
    Piece::WSilverX,
    Piece::WBishopX,
    Piece::WRookX,
    Piece::WPawnX,
    Piece::BKing,
    Piece::BGold,
    Piece::BSilver,
    Piece::BBishop,
    Piece::BRook,
    Piece::BPawn,
    Piece::BSilverX,
    Piece::BBishopX,
    Piece::BRookX,
    Piece::BPawnX,
];
pub const PIECE_TYPE_ALL: [PieceType; 10] = [
    PieceType::King,
    PieceType::Gold,
    PieceType::Silver,
    PieceType::Bishop,
    PieceType::Rook,
    PieceType::Pawn,
    PieceType::SilverX,
    PieceType::BishopX,
    PieceType::RookX,
    PieceType::PawnX,
];
pub const HAND_PIECE_TYPE_ALL: [PieceType; 5] =
    [PieceType::Gold, PieceType::Silver, PieceType::Bishop, PieceType::Rook, PieceType::Pawn];
pub const DIRECTION_ALL: [Direction; 8] = [
    Direction::N,
    Direction::NE,
    Direction::E,
    Direction::SE,
    Direction::S,
    Direction::SW,
    Direction::W,
    Direction::NW,
];

pub const SQUARE_NB: usize = 5 * 5;
pub const MAX_PLY: usize = 512;
