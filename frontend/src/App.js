import React, {useState, useRef, useEffect} from 'react';
import styled, {keyframes} from 'styled-components';
import 'bootstrap/dist/css/bootstrap.min.css';

import LoadingButton from './components/LoadingButton';
import NewGameModal from './components/NewGameModal';

const View = styled.div`
display: flex;
background-color: #283747;
min-height: 100vh;
flex-direction: column;
justify-content: center;
align-items: center;
color: white;
`;

const Header = styled.div`

`;

const Title = styled.div`
color: white;
font-size: 36px;
font-weight: 600;
`;

const GridContainer = styled.div`
/* game is 7cols x 6rows */
display: grid;
grid-template-rows: repeat(5, 1fr);
grid-template-columns: repeat(6, 1fr);
margin: 4em;
margin-top: 1em;
`;

const GameContainer = styled.div`
margin-top: 2em;
p {
  padding-left: 4em;
  padding-right: 4em;
}
`;

const CellContainer = styled.div`
grid-column-start: ${props => props.col};
grid-column-end: ${props => props.col + 1};
grid-row-start: ${props => props.row};
grid-row-end: ${props => props.row + 1};
display: flex;
justify-content: center;
align-items: center;
padding: .2em;
`;

const CellDot = styled.div`
  border-radius: 50%;
  background-color: ${props => props.val === 0 ? "#283747" : props.val === 1 ? "#45B39D" : "#C0392B" };
  border: 4px solid #5D6D7E;
  padding: 1.6em;
  cursor: ${props => props.placable ? 'pointer' : 'default'};
  :hover {
    background-color: ${props => props.placable ? (props.playerTurn === 1 ? "#45B39D" : "#C0392B") : null};
  };
  :active {
    background-color: ${props => props.placable ? (props.playerTurn === 1 ? "#378F7E" : "#9A2E22") : null};
  }
  animation-name: ${props => props.isWinningPiece ? blinkingEffect : null};
  animation-duration: 1s;
  animation-iteration-count: infinite;
  animation-timing-function: linear;
  animation-name: ${props => props.lastPlaced ? blinkingEffect1 : null};
`;

const blinkingEffect = () => {
  return keyframes`
  50% {
    border-color: white; 
  }`;
};

const blinkingEffect1 = () => {
  return keyframes`
  50% {
    opacity: 70%; 
  }`;
};

const bouncingText = () => {
  return keyframes`
  50% {
    transform: scale(1.5);
  }
  `
}

const WinningText = styled.p`
  margin: 0;
  padding-left: 4em;
  padding-right: 4em;
  width: fit-content;
  animation: ${bouncingText} 1s infinite forwards;
`;

const Cell = (props) => {
  return (
    <CellContainer col={props.col}  row={props.row}
    >
      <CellDot 
        onClick={props.placable ? props.handleClick : null}
        isWinningPiece={props.isWinningPiece}
        placable={props.placable}
        lastPlaced={props.lastPlaced}
        playerTurn={props.playerTurn}
        val={props.val}
      />
    </CellContainer>
  )

}

const App = () => {

  const [modalShow, setModalShow] = useState(false);
  const LoadingButtonRef = useRef(null);
  const [humanPlayer, setHumanPlayer] = useState(1);
  const [aiPlayer, setAIPlayer] = useState(2);
  const [playerTurn, setPlayerTurn] = useState(1);
  const [lastPlaced, setLastPlaced] = useState([-1,-1]);

  // 1: human, 2: ai
  const generateNewBoard = () => {
    return [[0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0]]; 
  }

  const [cells, setCells] = useState(generateNewBoard());
  const [winner, setWinner] = useState(0);
  const [winningLocations, setWinningLocations] = useState(generateNewBoard());
  const [gameStarted, setGameStarted] = useState(false);
  const [assist, setAssist] = useState(false);
  const playerNames = ['Human', 'Dense Agent', 'Alpha0 MCTS Agent', 'Alpha0 Greedy Agent', 'Negamax Agent', 'Random Agent']
  const [playerOneName, setPlayerOneName] = useState('');
  const [playerTwoName, setPlayerTwoName] = useState('');

  const applyGravity = (cells, x, y) => {
    for (let i = cells.length - 1; i >= x; i--){
      if (cells[i][y] === 0) {
        cells[i][y] = cells[x][y];
        cells[x][y] = 0;
      }
    }
    return cells;
  }

  const assistMove = (board) => {
    let promise = new Promise((resolve, reject) => {
      fetch('http://localhost:5000/getpiece')
      .then((response) => response.json())
      .then(async (data) => {
        resolve(data.action);
      })
      .catch((error) => {
        reject(error);
      })
    });
    promise.then((action) => {
      console.log('got action');
      let newCells = Array.from(board);
      newCells[0][action] = humanPlayer;
      applyGravity(newCells, 0, action);
      setCells(newCells);
      updateLastPlaced(board,newCells);
      setPlayerTurn(aiPlayer);
      setPiece(newCells,action);
    }, (error) => {console.log(error)});
  }

  const placeCell = (i,j) => {
    console.log(`place cell at row: ${i}, col: ${j}`);
    let newCells = Array.from(cells);
    newCells[i][j] = humanPlayer;
    applyGravity(newCells, i, j);
    setCells(newCells);
    setPlayerTurn(aiPlayer);
    setPiece(newCells,j);
  }
  
  const randomTimeout = (min,max) => {
    let delay = Math.floor(Math.random() * (max - min)) + min;
    return new Promise((resolve) => setTimeout(resolve, delay));
  }
  
  const chooseFirstMover = async (players) => {
    players = players.map((i)=>parseInt(i));
    let humanPosition = players.findIndex((i)=> i === 1);
    console.log(players);
    let aiPosition = -1;
    let assistAgent = -1;
    let opponentAgent = -1;
    if (humanPosition === -1){
      setAssist(true);
      assistAgent = players[0];
      opponentAgent = players[1];
      humanPosition = 1;
      aiPosition = 2;
    } else {
      aiPosition = humanPosition === 0 ? 2 : 1;
      opponentAgent = players[aiPosition - 1];
      humanPosition += 1;
    };
    setCells(generateNewBoard());
    setLastPlaced([-1,-1]);
    if (humanPosition === 1){
      setPlayerOneName(playerNames[assistAgent === -1 ? 0 : assistAgent - 1]);
      setPlayerTwoName(playerNames[opponentAgent - 1]);
    } else {
      setPlayerTwoName(playerNames[assistAgent === -1 ? 0 : assistAgent - 1]);
      setPlayerOneName(playerNames[opponentAgent - 1]);
    }
    let promise1 = new Promise((resolve) => {
      setHumanPlayer(humanPosition);
      setAIPlayer(humanPosition === 1 ? 2 : 1);
      setModalShow(false);
      resolve();
    });
    let promise = new Promise((resolve, reject) => {
      const requestOptions ={
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(
          {'opponent_agent': opponentAgent,'player': humanPosition, 'assist_agent': assistAgent}
        )
      };
      fetch('http://localhost:5000/createboard', requestOptions)
      .then(response => {response.json()})
      .then(() => {
        LoadingButtonRef.current.doneLoading();
        resolve();
      }).catch((error) => {
        reject(error);
      });
    });
    Promise.all([promise, promise1]).then(() => {
      console.log('successfully started new game!');
      getBoard(generateNewBoard());
      setWinningLocations(generateNewBoard());
      // setModalShow(false);
      setWinner(0);
      setGameStarted(true);
    }, (error)=>{console.log(error)});
  };

  const getBoard = (board) => {
    let promise = new Promise((resolve, reject) => {
      fetch('http://localhost:5000/getboard')
      .then((response) => response.json())
      .then(async (data) => {
        // await randomTimeout(100,1000);
        console.log(data);
        setCells(data.board);
        setPlayerTurn(data.mark);
        if (data.playing !== "playing") {
          if (data.playing === "draw"){
            setGameStarted(false);
            setWinner(3);
          } else {
            checkWinCondition(data.board);
          }
        } else {
          updateLastPlaced(board,data.board);
        }
        resolve(data);
      })
      .catch((error) => {
        reject(error);
      })
    });
    promise.then((data) => {
      console.log('got board');
      console.log(data);
      if (data.playing === 'playing' && data.assist){
        assistMove(data.board);
      }
    }, (error) => {console.log(error)});
  }

  const setPiece = (newCells, col) => {
    let promise = new Promise((resolve, reject) => {
      const requestOptions ={
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(
          {'action': col}
        )
      };
      fetch('http://localhost:5000/setpiece', requestOptions)
      .then(response => response.json())
      .then((data) => {
        resolve();
      }).catch((error) => {
        reject(error);
      });
    });
    promise.then(()=>{
      getBoard(newCells);
      console.log('submitted action');
    },
    (error)=>{console.log(error);})
  }

  const checkWinner = (cells) => {
    // adapted from https://stackoverflow.com/questions/33181356/connect-four-game-checking-for-wins-js
    const checkLine = (a,b,c,d) => {
      return ((a!==0)&&(a===b)&&(a===c)&&(a===d))
    }
    // check down
    for (let r=0; r < 3; r++){
      for (let c=0; c < 7; c++){
        if (checkLine(cells[r][c],cells[r+1][c],cells[r+2][c],cells[r+3][c])){
          return [cells[r][c], [[r,c],[r+1,c],[r+2,c],[r+3,c]]];
        }
      }
    }
    // check right
    for (let r=0; r < 6; r++){
      for (let c=0; c < 4; c++){
        if (checkLine(cells[r][c], cells[r][c+1], cells[r][c+2], cells[r][c+3])){
          return [cells[r][c], [[r,c],[r,c+1],[r,c+2],[r,c+3]]];
        }
      }
    }
    // check down-right
    for (let r=0; r < 3; r++){
      for (let c=0; c < 4; c++){
        if (checkLine(cells[r][c], cells[r+1][c+1], cells[r+2][c+2], cells[r+3][c+3])){
          return [cells[r][c], [[r,c],[r+1,c+1],[r+2,c+2],[r+3,c+3]]];
        }
      }
    }
    // check down-left
    for (let r=3; r < 6; r++){
      for (let c=0; c < 4; c++){
        if (checkLine(cells[r][c], cells[r-1][c+1], cells[r-2][c+2], cells[r-3][c+3])){
          return [cells[r][c], [[r,c],[r-1,c+1],[r-2,c+2],[r-3,c+3]]];
        }
      }
    }
    return [0,[]];
  }

  const checkWinCondition = (board) => {
    let [winner, locations] = checkWinner(board);
    if (winner !== 0) {
      console.log('winner found!');
      let newWinningCells = generateNewBoard();
      for (let k=0; k < locations.length; k++){
        newWinningCells[locations[k][0]][locations[k][1]] = 1;
      }
      setLastPlaced([-1,-1]);
      setWinningLocations(newWinningCells);
      setWinner(winner);
      setGameStarted(false);
    } else {
      setWinningLocations(generateNewBoard());
      setWinner(0);
    }
  }

  const updateLastPlaced = (oldCells, newCells) => {
    console.log(oldCells);
    console.log(newCells);
    for (let i=0; i < newCells.length; i++){
      for (let j=0; j < newCells[0].length; j++){
        if (newCells[i][j] !== oldCells[i][j]){
          setLastPlaced([i,j])
          console.log(`AI placed at row: ${i}, col: ${j}`)
        }
      }
    }
  }

  return (
    <>
      <View>
        <Header>
          <Title>Connect-X with RL</Title>
        </Header>
        <GameContainer>
          {(gameStarted || winner !== 0) ? <p style={{textAlign: 'center'}}><span style={{color: '#45B39D', fontWeight: '700'}}>{playerOneName}</span> vs <span style={{color: '#C0392B', fontWeight: '700'}}>{playerTwoName}</span></p> : <p style={{textAlign: 'center'}}>Hello, please start a new game!</p>}
          {gameStarted ? 
            <p>Turn: {playerTurn === 1 ? <span style={{color: '#45B39D', fontWeight: '700'}}>{playerOneName}</span> : <span style={{color: '#C0392B', fontWeight: '700'}}>{playerTwoName}</span>}</p>
            :
            winner === 0 ? 
            <p style={{opacity: 0}}>Turn: </p>
            :
            <div>
              {winner === 3 ?
              <WinningText>Draw!</WinningText>
              :
              <WinningText>Winner: {winner === 1 ? <span style={{color: '#45B39D', fontWeight: '700'}}>{playerOneName}</span> : <span style={{color: '#C0392B', fontWeight: '700'}}>{playerTwoName}</span>}</WinningText>}
            </div>
          }
          <GridContainer>
            {cells.map((row, i) => 
              row.map((cell, j) => (
              <Cell 
                key={i*10 + j}
                col={j}
                row={i}
                val={cell}
                playerTurn={playerTurn}
                lastPlaced={lastPlaced[0] === i && lastPlaced[1] === j}
                // placable={cell === 0 && gameStarted}
                isWinningPiece = {winningLocations[i][j] === 1}
                placable={playerTurn === humanPlayer && cell === 0 && gameStarted && !assist}
                handleClick={assist ? () => {return} : () => {placeCell(i,j)}}
                />
              )))
            }
          </GridContainer>
        </GameContainer>
        <LoadingButton
          variant="secondary"
          label="New Game"
          loadingLabel="Loading New Game..."
          gameStarted={gameStarted}
          ref={LoadingButtonRef}
          handleClick={() => {
            setModalShow(true)}}/>
      </View>
      <NewGameModal show={modalShow}
        onHide={()=> {
          setModalShow(false);
          LoadingButtonRef.current.doneLoading();}
        }
        handleClick={chooseFirstMover}/>
    </>
  );
}

export default App;
