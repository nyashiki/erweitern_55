#[derive(PartialEq)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1
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

#[derive(PartialEq)]
#[repr(u8)]
pub enum Piece {
    NoPiece = 0,

    WKing = 0b00001, WGold = 0b00010, WSilver  = 0b00011, WBishop  = 0b00100, WRook  = 0b00101, WPawn  = 0b00110,
                                      WSilverX = 0b01011, WBishopX = 0b01100, WRookX = 0b01101, WPawnX = 0b01110,

    BKing = 0b10001, BGold = 0b10010, BSilver  = 0b10011, BBishop  = 0b10100, BRook  = 0b10101, BPawn  = 0b10110,
                                      BSilverX = 0b11011, BBishopX = 0b11100, BRookX = 0b11101, BPawnX = 0b11110,
}

impl Piece {
    pub fn get_promoted(self) -> Piece {
        match self {
            Piece::WSilver => Piece::WSilverX,
            Piece::WBishop => Piece::WBishopX,
            Piece::WRook   => Piece::WRookX,
            Piece::WPawn   => Piece::WPawnX,
            Piece::BSilver => Piece::BSilverX,
            Piece::BBishop => Piece::BBishopX,
            Piece::BRook   => Piece::BRookX,
            Piece::BPawn   => Piece::BPawnX,
            _              => Piece::NoPiece
        }
    }

    pub fn is_promoted(self) -> bool {
        match self {
            Piece::WSilverX => true,
            Piece::WBishopX => true,
            Piece::WRookX   => true,
            Piece::WPawnX   => true,
            Piece::BSilverX => true,
            Piece::BBishopX => true,
            Piece::BRookX   => true,
            Piece::BPawnX   => true,
            _               => false
        }
    }

    pub fn get_raw(self) -> Piece {
        match self {
            Piece::WSilverX => Piece::WSilver,
            Piece::WBishopX => Piece::WBishop,
            Piece::WRookX   => Piece::WRook,
            Piece::WPawnX   => Piece::WPawn,
            Piece::BSilverX => Piece::BSilver,
            Piece::BBishopX => Piece::BBishop,
            Piece::BRookX   => Piece::BRook,
            Piece::BPawnX   => Piece::BPawn,
            _               => Piece::NoPiece
        }
    }

    pub fn is_raw(self) -> bool {
        !self.is_promoted()
    }
}

#[derive(PartialEq)]
#[repr(u8)]
pub enum PieceType {
    NoPieceType = 0,

    King = 0b0001, Gold = 0b0010, Silver  = 0b0011, Bishop  = 0b0100, Rook  = 0b0101, Pawn  = 0b0110,
                                  SilverX = 0b1011, BishopX = 0b1100, RookX = 0b1101, PawnX = 0b1110,
}

impl PieceType {
    pub fn get_promoted(self) -> PieceType {
        match self {
            PieceType::Silver => PieceType::SilverX,
            PieceType::Bishop => PieceType::BishopX,
            PieceType::Rook   => PieceType::RookX,
            PieceType::Pawn   => PieceType::PawnX,
            _                 => PieceType::NoPieceType
        }
    }

    pub fn is_promoted(self) -> bool {
        match self {
            PieceType::SilverX => true,
            PieceType::BishopX => true,
            PieceType::RookX   => true,
            PieceType::PawnX   => true,
            _                  => false
        }
    }

    pub fn get_raw(self) -> PieceType {
        match self {
            PieceType::SilverX => PieceType::Silver,
            PieceType::BishopX => PieceType::Bishop,
            PieceType::RookX   => PieceType::Rook,
            PieceType::PawnX   => PieceType::Pawn,
            _                  => PieceType::NoPieceType
        }
    }

    pub fn is_raw(self) -> bool {
        !self.is_promoted()
    }
}

#[test]
fn get_promoted_test() {
    // Piece
    assert!(Piece::WSilver.get_promoted() == Piece::WSilverX);
    assert!(Piece::WBishop.get_promoted() == Piece::WBishopX);
    assert!(Piece::WRook.get_promoted()   == Piece::WRookX);
    assert!(Piece::WPawn.get_promoted()   == Piece::WPawnX);
    assert!(Piece::BSilver.get_promoted() == Piece::BSilverX);
    assert!(Piece::BBishop.get_promoted() == Piece::BBishopX);
    assert!(Piece::BRook.get_promoted()   == Piece::BRookX);
    assert!(Piece::BPawn.get_promoted()   == Piece::BPawnX);

    // PieceType
    assert!(PieceType::Silver.get_promoted() == PieceType::SilverX);
    assert!(PieceType::Bishop.get_promoted() == PieceType::BishopX);
    assert!(PieceType::Rook.get_promoted()   == PieceType::RookX);
    assert!(PieceType::Pawn.get_promoted()   == PieceType::PawnX);
}

#[test]
fn is_promoted_test() {
    // Piece
    assert!(Piece::WSilverX.is_promoted());
    assert!(Piece::WBishopX.is_promoted());
    assert!(Piece::WRookX.is_promoted());
    assert!(Piece::WPawnX.is_promoted());
    assert!(Piece::BSilverX.is_promoted());
    assert!(Piece::BBishopX.is_promoted());
    assert!(Piece::BRookX.is_promoted());
    assert!(Piece::BPawnX.is_promoted());

    // PieceType
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
    assert!(Piece::WRookX.get_raw()   == Piece::WRook);
    assert!(Piece::WPawnX.get_raw()   == Piece::WPawn);
    assert!(Piece::BSilverX.get_raw() == Piece::BSilver);
    assert!(Piece::BBishopX.get_raw() == Piece::BBishop);
    assert!(Piece::BRookX.get_raw()   == Piece::BRook);
    assert!(Piece::BPawnX.get_raw()   == Piece::BPawn);

    // PieceType
    assert!(PieceType::SilverX.get_raw() == PieceType::Silver);
    assert!(PieceType::BishopX.get_raw() == PieceType::Bishop);
    assert!(PieceType::RookX.get_raw()   == PieceType::Rook);
    assert!(PieceType::PawnX.get_raw()   == PieceType::Pawn);
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
