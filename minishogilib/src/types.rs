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
